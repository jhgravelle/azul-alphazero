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
    Convolutions over this channel learn adjacency implicitly.

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
    Range [0, 1]. Already-filled cells get a value reflecting how
    developed that region of the wall is.

  Bag count (ch 6):
    bag_tiles_of_color / 20, broadcast across all rows.
    col determines color via COLOR_TILES[col].

  Source distribution (ch 7):
    row = bucket index 0–4 (bucket b = sources with exactly b+1 tiles
          of this color; bucket 4 = sources with 5+ tiles)
    col = color index (COLOR_TILES[col])
    value = number of sources in that bucket / 5
    Max simultaneous sources = 5.

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

Move index layout (unchanged from v1)
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

from engine.game import Game, Move, CENTER, FLOOR
from engine.constants import (
    BOARD_SIZE,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
    PLAYERS,
    TILES_PER_COLOR,
    WALL_PATTERN,
    Tile,
)
from engine.scoring import (
    earned_score_unclamped,
    score_floor_penalty,
    pending_placement_details,
)

# ── Move-space constants ───────────────────────────────────────────────────

NUM_FACTORIES: int = 2 * PLAYERS + 1
NUM_SOURCES: int = NUM_FACTORIES + 1  # factories + center
NUM_DESTINATIONS: int = BOARD_SIZE + 1  # pattern lines + floor
MOVE_SPACE_SIZE: int = NUM_SOURCES * BOARD_SIZE * NUM_DESTINATIONS

# ── Spatial constants ──────────────────────────────────────────────────────

NUM_COLORS: int = len(COLOR_TILES)  # 5
NUM_CHANNELS: int = 8
SPATIAL_SHAPE: tuple = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)  # (8, 5, 5)

# Channel indices — named constants so helpers are self-documenting
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
FLOOR_PENALTY_DIVISOR: float = 14.0  # max possible floor penalty
MAX_BAG_TILES: float = float(TILES_PER_COLOR)  # 20
MAX_SOURCES: float = 5.0  # max simultaneous sources with tiles
SOURCE_DIST_BUCKETS: int = 5  # buckets: 1, 2, 3, 4, 5+ tiles

# ── Internal helpers — wall geometry ──────────────────────────────────────


def _build_color_for_wall_cell() -> list[list[Tile]]:
    """Return color_for_wall_cell[row][wall_col] = Tile at that wall position."""
    return [
        [WALL_PATTERN[row][wall_col] for wall_col in range(BOARD_SIZE)]
        for row in range(BOARD_SIZE)
    ]


# Precomputed once at import time — avoids repeated WALL_PATTERN lookups
_COLOR_FOR_WALL_CELL: list[list[Tile]] = _build_color_for_wall_cell()


def _count_filled_in_rows(wall: list[list[Tile | None]]) -> list[int]:
    """Return filled cell count for each wall row."""
    return [
        sum(1 for wall_col in range(BOARD_SIZE) if wall[row][wall_col] is not None)
        for row in range(BOARD_SIZE)
    ]


def _count_filled_in_cols(wall: list[list[Tile | None]]) -> list[int]:
    """Return filled cell count for each wall column."""
    return [
        sum(1 for row in range(BOARD_SIZE) if wall[row][wall_col] is not None)
        for wall_col in range(BOARD_SIZE)
    ]


def _count_filled_per_color(wall: list[list[Tile | None]]) -> dict[Tile, int]:
    """Return how many wall cells are filled for each color."""
    counts: dict[Tile, int] = {color: 0 for color in COLOR_TILES}
    for row in range(BOARD_SIZE):
        for wall_col in range(BOARD_SIZE):
            if wall[row][wall_col] is not None:
                color = _COLOR_FOR_WALL_CELL[row][wall_col]
                counts[color] += 1
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
    board,
    wall: list[list[Tile | None]],
) -> None:
    """Write pattern line fill ratio into the committed color's wall column.

    For each row, if the pattern line has a committed color and the
    corresponding wall cell is not yet filled, write fill_ratio at
    (row, wall_col_for_that_color). All other cells in the row stay 0.0.
    """
    for row in range(BOARD_SIZE):
        line = board.pattern_lines[row]
        if not line:
            continue
        committed_color = line[0]
        wall_col = COLUMN_FOR_TILE_IN_ROW[committed_color][row]
        if wall[row][wall_col] is not None:
            # Wall cell already filled — no pending placement signal needed
            continue
        capacity = row + 1
        fill_ratio = len(line) / capacity
        spatial[channel, row, wall_col] = fill_ratio


def _encode_bonus_proximity(
    spatial: torch.Tensor,
    channel: int,
    post_placement_wall: list[list[Tile | None]],
) -> None:
    """Write bonus proximity into each wall cell.

    Uses the post-placement wall (pattern lines that complete this round
    are already applied) so the signal reflects end-of-round state.

    For each cell (row, wall_col):
      row_progress   = filled_in_row / 5
      col_progress   = filled_in_col / 5
      color_progress = filled_of_this_color / 5
      value = (row_progress + col_progress + color_progress) / 3
    """
    filled_in_rows = _count_filled_in_rows(post_placement_wall)
    filled_in_cols = _count_filled_in_cols(post_placement_wall)
    filled_per_color = _count_filled_per_color(post_placement_wall)

    for row in range(BOARD_SIZE):
        row_progress = filled_in_rows[row] / BOARD_SIZE
        for wall_col in range(BOARD_SIZE):
            col_progress = filled_in_cols[wall_col] / BOARD_SIZE
            color = _COLOR_FOR_WALL_CELL[row][wall_col]
            color_progress = filled_per_color[color] / BOARD_SIZE
            spatial[channel, row, wall_col] = (
                row_progress + col_progress + color_progress
            ) / 3.0


def _encode_bag_count(
    spatial: torch.Tensor,
    channel: int,
    game: Game,
) -> None:
    """Write bag tile count per color, broadcast across all rows.

    col = color index in COLOR_TILES order.
    value = bag_count / 20.
    """
    for color_idx, color in enumerate(COLOR_TILES):
        count = game.state.bag.count(color) / MAX_BAG_TILES
        for row in range(BOARD_SIZE):
            spatial[channel, row, color_idx] = count


def _encode_source_distribution(
    spatial: torch.Tensor,
    channel: int,
    game: Game,
) -> None:
    """Write source distribution into ch 7.

    row = bucket index (0 = sources with 1 tile, 4 = sources with 5+ tiles)
    col = color index in COLOR_TILES order
    value = number of sources in that bucket / 5

    Counts tiles across all factories and the center pile.
    """
    for color_idx, color in enumerate(COLOR_TILES):
        # Gather per-source counts for this color
        source_counts: list[int] = []
        for factory in game.state.factories:
            count = factory.count(color)
            if count > 0:
                source_counts.append(count)
        center_count = game.state.center.count(color)
        if center_count > 0:
            source_counts.append(center_count)

        # Fill buckets
        bucket_counts = [0] * SOURCE_DIST_BUCKETS
        for source_count in source_counts:
            bucket_index = min(source_count - 1, SOURCE_DIST_BUCKETS - 1)
            bucket_counts[bucket_index] += 1

        for bucket_index, sources_in_bucket in enumerate(bucket_counts):
            spatial[channel, bucket_index, color_idx] = sources_in_bucket / MAX_SOURCES


# ── Internal helpers — flat encoders ──────────────────────────────────────


def _encode_flat_scores(
    flat: torch.Tensor,
    my_board,
    opp_board,
) -> None:
    """Write official scores and earned-this-round unclamped scores."""
    flat[OFF_MY_SCORE] = my_board.score / MAX_SCORE_DIVISOR
    flat[OFF_OPP_SCORE] = opp_board.score / MAX_SCORE_DIVISOR
    flat[OFF_MY_EARNED] = earned_score_unclamped(my_board) / EARNED_DIVISOR
    flat[OFF_OPP_EARNED] = earned_score_unclamped(opp_board) / EARNED_DIVISOR


def _encode_flat_floor_penalties(
    flat: torch.Tensor,
    my_board,
    opp_board,
) -> None:
    """Write floor penalty values (negative) normalized by max penalty."""
    flat[OFF_MY_FLOOR] = (
        score_floor_penalty(my_board.floor_line) / FLOOR_PENALTY_DIVISOR
    )
    flat[OFF_OPP_FLOOR] = (
        score_floor_penalty(opp_board.floor_line) / FLOOR_PENALTY_DIVISOR
    )


def _encode_flat_first_player_tokens(
    flat: torch.Tensor,
    my_board,
    opp_board,
) -> None:
    """Write first-player token flags for each player."""
    flat[OFF_MY_FP_TOKEN] = 1.0 if Tile.FIRST_PLAYER in my_board.floor_line else 0.0
    flat[OFF_OPP_FP_TOKEN] = 1.0 if Tile.FIRST_PLAYER in opp_board.floor_line else 0.0


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
    current_player = game.state.current_player
    opponent = 1 - current_player
    my_board = game.state.players[current_player]
    opp_board = game.state.players[opponent]

    # Post-placement walls — used for bonus proximity and pattern channel
    # pending_placement_details simulates end-of-round scoring without mutating
    _my_details, my_post_wall = pending_placement_details(my_board)
    _opp_details, opp_post_wall = pending_placement_details(opp_board)

    # ── Spatial ───────────────────────────────────────────────────────────
    spatial = torch.zeros(SPATIAL_SHAPE, dtype=torch.float32)

    _encode_wall_filled(spatial, CH_MY_WALL, my_board.wall)
    _encode_pattern_line_fill_ratio(spatial, CH_MY_PATTERN, my_board, my_board.wall)
    _encode_bonus_proximity(spatial, CH_MY_BONUS, my_post_wall)

    _encode_wall_filled(spatial, CH_OPP_WALL, opp_board.wall)
    _encode_pattern_line_fill_ratio(spatial, CH_OPP_PATTERN, opp_board, opp_board.wall)
    _encode_bonus_proximity(spatial, CH_OPP_BONUS, opp_post_wall)

    _encode_bag_count(spatial, CH_BAG, game)
    _encode_source_distribution(spatial, CH_SOURCE_DIST, game)

    # ── Flat ──────────────────────────────────────────────────────────────
    flat = torch.zeros(FLAT_SIZE, dtype=torch.float32)

    _encode_flat_scores(flat, my_board, opp_board)
    _encode_flat_floor_penalties(flat, my_board, opp_board)
    _encode_flat_first_player_tokens(flat, my_board, opp_board)

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
