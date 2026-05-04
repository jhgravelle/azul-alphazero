# neural/encoder.py

"""Encodes Azul game states and moves as tensors for the neural network.

Spatial tensor: shape (8, 5, 5)
---------------------------------------------------------------------
All channels use (row, wall_col) space where:
  row     = pattern line row 0–4
  wall_col = wall column 0–4

The wall pattern means each (row, wall_col) cell corresponds to exactly
one color, given by WALL_PATTERN[row][wall_col].

Channel layout:
  0  My wall filled
  1  My pattern line fill ratio
  2  My bonus proximity
  3  Opponent wall filled
  4  Opponent pattern line fill ratio
  5  Opponent bonus proximity
  6  Bag count by color  (broadcast across rows)
  7  Source distribution (row = bucket, col = color)

Channel details:

  Wall filled (ch 0, 3):
    1.0 if that wall cell is occupied, 0.0 otherwise.

  Pattern line fill ratio (ch 1, 4):
    For each row, exactly one wall_col is nonzero — the column that
    corresponds to the committed color for that pattern line.
    Value = filled_tiles / capacity (0.0 if line is empty or wall
    cell already filled).

  Bonus proximity (ch 2, 5):
    Computed on the post-placement wall (pattern lines that will
    complete this round are treated as already placed).
    For each cell (row, wall_col):
      row_progress   = filled cells in this wall row / 5
      col_progress   = filled cells in this wall col / 5
      color_progress = filled cells of WALL_PATTERN[row][wall_col] / 5
      value = (row_progress + col_progress + color_progress) / 3
    Range [0, 1].

  Bag count (ch 6):
    bag_tiles_of_color / 20, broadcast across all rows.
    col determines color via COLOR_TILES[col].

  Source distribution (ch 7):
    row = bucket index 0–4 (bucket b = sources with exactly b+1 tiles
          of this color; bucket 4 = sources with 5+ tiles)
    col = color index (COLOR_TILES[col])
    value = number of sources in that bucket / 5

Flat vector: shape (FLAT_SIZE,) = (8,)
---------------------------------------------------------------------
Offset  Name
------  ----
  0     My official score / 100
  1     Opponent official score / 100
  2     My earned-this-round unclamped / 50
  3     Opponent earned-this-round unclamped / 50
  4     My floor penalty / 14   (negative value, so range [-1, 0])
  5     Opponent floor penalty / 14
  6     I hold first-player token (0 or 1)
  7     Opponent holds first-player token (0 or 1)

Move index layout
-----------------------------------------------------
Each move is a triple (source_idx, color_idx, dest_idx):
  source_idx : 0..NUM_FACTORIES-1 = factory;  NUM_FACTORIES = center
  color_idx  : 0..BOARD_SIZE-1  (COLOR_TILES order)
  dest_idx   : 0..BOARD_SIZE-1 = pattern line;  BOARD_SIZE = floor

Flat index = source_idx * (BOARD_SIZE * NUM_DESTINATIONS)
           + color_idx  * NUM_DESTINATIONS
           + dest_idx

MOVE_SPACE_SIZE = NUM_SOURCES * BOARD_SIZE * NUM_DESTINATIONS
"""

import torch

from engine.constants import (
    BOARD_SIZE,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
    CUMULATIVE_FLOOR_PENALTIES,
    PLAYERS,
    TILES_PER_COLOR,
    WALL_PATTERN,
    Tile,
)
from engine.game import Game, Move, CENTER, FLOOR
from engine.player import Player

# ── Move-space constants ───────────────────────────────────────────────────

NUM_FACTORIES: int = 2 * PLAYERS + 1
NUM_SOURCES: int = NUM_FACTORIES + 1  # factories + center
NUM_DESTINATIONS: int = BOARD_SIZE + 1  # pattern lines + floor
MOVE_SPACE_SIZE: int = NUM_SOURCES * BOARD_SIZE * NUM_DESTINATIONS

# ── Spatial constants ──────────────────────────────────────────────────────

NUM_COLORS: int = len(COLOR_TILES)  # 5
NUM_CHANNELS: int = 8
SPATIAL_SHAPE: tuple = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)  # (8, 5, 5)

CH_MY_WALL: int = 0
CH_MY_PATTERN: int = 1
CH_MY_BONUS: int = 2
CH_OPP_WALL: int = 3
CH_OPP_PATTERN: int = 4
CH_OPP_BONUS: int = 5
CH_BAG: int = 6
CH_SOURCE_DIST: int = 7

# ── Flat constants ─────────────────────────────────────────────────────────

OFF_MY_SCORE: int = 0
OFF_OPP_SCORE: int = 1
OFF_MY_EARNED: int = 2
OFF_OPP_EARNED: int = 3
OFF_MY_FLOOR: int = 4
OFF_OPP_FLOOR: int = 5
OFF_MY_FP_TOKEN: int = 6
OFF_OPP_FP_TOKEN: int = 7
FLAT_SIZE: int = 8

# ── Normalization constants ────────────────────────────────────────────────

MAX_SCORE_DIVISOR: float = 100.0
EARNED_DIVISOR: float = 50.0
FLOOR_PENALTY_DIVISOR: float = 14.0
MAX_BAG_TILES: float = float(TILES_PER_COLOR)  # 20
MAX_SOURCES: float = 5.0
SOURCE_DIST_BUCKETS: int = 5

# ── Internal helpers — wall geometry ──────────────────────────────────────


def _build_color_for_wall_cell() -> list[list[Tile]]:
    """Return color_for_wall_cell[row][wall_col] = Tile at that wall position."""
    return [
        [WALL_PATTERN[row][wall_col] for wall_col in range(BOARD_SIZE)]
        for row in range(BOARD_SIZE)
    ]


_COLOR_FOR_WALL_CELL: list[list[Tile]] = _build_color_for_wall_cell()


def _build_post_placement_wall(player: Player) -> list[list[Tile | None]]:
    """Return a copy of the wall with all pending full pattern lines placed.

    Simulates end-of-round tile placement in row order so that adjacency
    scores are correct when pending placements are adjacent to each other.
    Does not mutate player.wall.
    """
    wall: list[list[Tile | None]] = [row[:] for row in player.wall]
    for row, line in enumerate(player.pattern_lines):
        if len(line) < row + 1:
            continue
        tile = line[0]
        col = COLUMN_FOR_TILE_IN_ROW[tile][row]
        wall[row][col] = tile
    return wall


def _count_filled_in_rows(wall: list[list[Tile | None]]) -> list[int]:
    """Return pattern-tile-weighted fill count for each wall row."""
    return [
        sum(
            row + 1 for wall_col in range(BOARD_SIZE) if wall[row][wall_col] is not None
        )
        for row in range(BOARD_SIZE)
    ]


def _count_filled_in_cols(wall: list[list[Tile | None]]) -> list[int]:
    """Return pattern-tile-weighted fill count for each wall column."""
    return [
        sum(row + 1 for row in range(BOARD_SIZE) if wall[row][wall_col] is not None)
        for wall_col in range(BOARD_SIZE)
    ]


def _count_filled_per_color(wall: list[list[Tile | None]]) -> dict[Tile, int]:
    """Return pattern-tile-weighted fill count for each color."""
    counts: dict[Tile, int] = {color: 0 for color in COLOR_TILES}
    for row in range(BOARD_SIZE):
        for wall_col in range(BOARD_SIZE):
            if wall[row][wall_col] is not None:
                color = _COLOR_FOR_WALL_CELL[row][wall_col]
                counts[color] += row + 1
    return counts


# ── Internal helpers — channel encoders ───────────────────────────────────


def _encode_wall_filled(
    spatial: torch.Tensor,
    channel: int,
    wall: list[list[Tile | None]],
) -> None:
    """Write 1.0 for each occupied wall cell into the given channel."""
    for row in range(BOARD_SIZE):
        for wall_col in range(BOARD_SIZE):
            if wall[row][wall_col] is not None:
                spatial[channel, row, wall_col] = 1.0


def _encode_pattern_line_fill_ratio(
    spatial: torch.Tensor,
    channel: int,
    player: Player,
    wall: list[list[Tile | None]],
) -> None:
    """Write pattern line fill ratio into the committed color's wall column."""
    for row in range(BOARD_SIZE):
        line = player.pattern_lines[row]
        if not line:
            continue
        committed_color = line[0]
        wall_col = COLUMN_FOR_TILE_IN_ROW[committed_color][row]
        if wall[row][wall_col] is not None:
            continue
        capacity = row + 1
        fill_ratio = len(line) / capacity
        spatial[channel, row, wall_col] = fill_ratio


def _add_partial_pattern_line_contributions(
    player: Player,
    filled_in_rows: list[int],
    filled_in_cols: list[int],
    filled_per_color: dict[Tile, int],
) -> None:
    """Add weighted tile counts from partial pattern lines to the progress totals."""
    for row in range(BOARD_SIZE):
        line = player.pattern_lines[row]
        capacity = row + 1
        if not line or len(line) == capacity:
            continue
        committed_color = line[0]
        wall_col = COLUMN_FOR_TILE_IN_ROW[committed_color][row]
        weighted_contribution = len(line) * (row + 1)
        filled_in_rows[row] += weighted_contribution
        filled_in_cols[wall_col] += weighted_contribution
        filled_per_color[committed_color] += weighted_contribution


def _encode_bonus_proximity(
    spatial: torch.Tensor,
    channel: int,
    player: Player,
    post_placement_wall: list[list[Tile | None]],
) -> None:
    """Write bonus proximity into each wall cell."""
    _MAX_WEIGHTED_SUM = 15

    filled_in_rows = _count_filled_in_rows(post_placement_wall)
    filled_in_cols = _count_filled_in_cols(post_placement_wall)
    filled_per_color = _count_filled_per_color(post_placement_wall)

    _add_partial_pattern_line_contributions(
        player, filled_in_rows, filled_in_cols, filled_per_color
    )

    for row in range(BOARD_SIZE):
        for wall_col in range(BOARD_SIZE):
            color = _COLOR_FOR_WALL_CELL[row][wall_col]
            spatial[channel, row, wall_col] = (
                (_MAX_WEIGHTED_SUM - filled_in_rows[row]) / _MAX_WEIGHTED_SUM
                + (_MAX_WEIGHTED_SUM - filled_in_cols[wall_col]) / _MAX_WEIGHTED_SUM
                + (_MAX_WEIGHTED_SUM - filled_per_color[color]) / _MAX_WEIGHTED_SUM
            ) / 3.0


def _encode_bag_count(
    spatial: torch.Tensor,
    channel: int,
    game: Game,
) -> None:
    """Write bag tile count per color, broadcast across all rows."""
    for color_idx, color in enumerate(COLOR_TILES):
        count = game.bag.count(color) / MAX_BAG_TILES
        for row in range(BOARD_SIZE):
            spatial[channel, row, color_idx] = count


def _encode_source_distribution(
    spatial: torch.Tensor,
    channel: int,
    game: Game,
) -> None:
    """Write source distribution into ch 7."""
    for color_idx, color in enumerate(COLOR_TILES):
        source_counts: list[int] = []
        for factory in game.factories:
            count = factory.count(color)
            if count > 0:
                source_counts.append(count)
        center_count = game.center.count(color)
        if center_count > 0:
            source_counts.append(center_count)

        bucket_counts = [0] * SOURCE_DIST_BUCKETS
        for source_count in source_counts:
            bucket_index = min(source_count - 1, SOURCE_DIST_BUCKETS - 1)
            bucket_counts[bucket_index] += 1

        for bucket_index, sources_in_bucket in enumerate(bucket_counts):
            spatial[channel, bucket_index, color_idx] = sources_in_bucket / MAX_SOURCES


# ── Internal helpers — flat encoders ──────────────────────────────────────


def _encode_flat_scores(
    flat: torch.Tensor,
    my_player: Player,
    opp_player: Player,
) -> None:
    """Write official scores and earned-this-round values."""
    flat[OFF_MY_SCORE] = my_player.score / MAX_SCORE_DIVISOR
    flat[OFF_OPP_SCORE] = opp_player.score / MAX_SCORE_DIVISOR
    flat[OFF_MY_EARNED] = my_player.earned / EARNED_DIVISOR
    flat[OFF_OPP_EARNED] = opp_player.earned / EARNED_DIVISOR


def _encode_flat_floor_penalties(
    flat: torch.Tensor,
    my_player: Player,
    opp_player: Player,
) -> None:
    """Write floor penalty values (negative) normalized by max penalty."""
    flat[OFF_MY_FLOOR] = (
        CUMULATIVE_FLOOR_PENALTIES[len(my_player.floor_line)] / FLOOR_PENALTY_DIVISOR
    )
    flat[OFF_OPP_FLOOR] = (
        CUMULATIVE_FLOOR_PENALTIES[len(opp_player.floor_line)] / FLOOR_PENALTY_DIVISOR
    )


def _encode_flat_first_player_tokens(
    flat: torch.Tensor,
    my_player: Player,
    opp_player: Player,
) -> None:
    """Write first-player token flags for each player."""
    flat[OFF_MY_FP_TOKEN] = 1.0 if Tile.FIRST_PLAYER in my_player.floor_line else 0.0
    flat[OFF_OPP_FP_TOKEN] = 1.0 if Tile.FIRST_PLAYER in opp_player.floor_line else 0.0


# ── Move index helpers ─────────────────────────────────────────────────────


def _source_to_idx(source: int) -> int:
    return NUM_FACTORIES if source == CENTER else source


def _idx_to_source(idx: int) -> int:
    return CENTER if idx == NUM_FACTORIES else idx


def _dest_to_idx(destination: int) -> int:
    return BOARD_SIZE if destination == FLOOR else destination


def _idx_to_dest(idx: int) -> int:
    return FLOOR if idx == BOARD_SIZE else idx


# ── Public API ─────────────────────────────────────────────────────────────


def encode_state(game: Game) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a Game into (spatial, flat) tensors.

    spatial : float32 tensor of shape SPATIAL_SHAPE = (8, 5, 5)
    flat    : float32 tensor of shape (FLAT_SIZE,) = (8,)
    """
    current_player_index = game.current_player_index
    opponent_index = 1 - current_player_index
    my_player = game.players[current_player_index]
    opp_player = game.players[opponent_index]

    my_post_wall = _build_post_placement_wall(my_player)
    opp_post_wall = _build_post_placement_wall(opp_player)

    spatial = torch.zeros(SPATIAL_SHAPE, dtype=torch.float32)

    _encode_wall_filled(spatial, CH_MY_WALL, my_player.wall)
    _encode_pattern_line_fill_ratio(spatial, CH_MY_PATTERN, my_player, my_player.wall)
    _encode_bonus_proximity(spatial, CH_MY_BONUS, my_player, my_post_wall)

    _encode_wall_filled(spatial, CH_OPP_WALL, opp_player.wall)
    _encode_pattern_line_fill_ratio(
        spatial, CH_OPP_PATTERN, opp_player, opp_player.wall
    )
    _encode_bonus_proximity(spatial, CH_OPP_BONUS, opp_player, opp_post_wall)

    _encode_bag_count(spatial, CH_BAG, game)
    _encode_source_distribution(spatial, CH_SOURCE_DIST, game)

    flat = torch.zeros(FLAT_SIZE, dtype=torch.float32)

    _encode_flat_scores(flat, my_player, opp_player)
    _encode_flat_floor_penalties(flat, my_player, opp_player)
    _encode_flat_first_player_tokens(flat, my_player, opp_player)

    return spatial, flat


def encode_move(move: Move, game: Game) -> int:
    """Encode a Move as a unique integer index in [0, MOVE_SPACE_SIZE)."""
    source_idx = _source_to_idx(move.source)
    color_idx = COLOR_TILES.index(move.tile)
    dest_idx = _dest_to_idx(move.destination)
    return (
        source_idx * (BOARD_SIZE * NUM_DESTINATIONS)
        + color_idx * NUM_DESTINATIONS
        + dest_idx
    )


def decode_move(index: int, game: Game) -> Move:
    """Decode an integer index back into a Move."""
    source_idx, remainder = divmod(index, BOARD_SIZE * NUM_DESTINATIONS)
    color_idx, dest_idx = divmod(remainder, NUM_DESTINATIONS)
    return Move(
        source=_idx_to_source(source_idx),
        tile=COLOR_TILES[color_idx],
        destination=_idx_to_dest(dest_idx),
    )
