# neural/encoder.py

"""Encodes Azul game states and moves as tensors for the neural network.

Spatial tensor: shape (4, 5, 5)
---------------------------------------------------------------------
All channels use (row, wall_col) space where:
  row      = pattern line row 0–4
  wall_col = wall column 0–4

Channel layout:
  0  My wall filled
  1  My pattern line fill ratio
  2  Opponent wall filled
  3  Opponent pattern line fill ratio

Channel details:

  Wall filled (ch 0, 2):
    1.0 if that wall cell is occupied, 0.0 otherwise.
    Uses the current wall (post-round confirmations only).

  Pattern line fill ratio (ch 1, 3):
    For each row, exactly one wall_col is nonzero — the column that
    corresponds to the committed color for that pattern line.
    Value = filled_tiles / capacity (0.0 if line is empty or wall
    cell already filled).

Flat vector: shape (FLAT_SIZE,) = (53,)
---------------------------------------------------------------------
Offset  Count  Name
------  -----  ----
  0       1    My official score / 100
  1       1    Opponent official score / 100
  2       1    My earned (unclamped) / 100
  3       1    Opponent earned (unclamped) / 100
  4       1    My floor penalty / 14   (negative, range [-1, 0])
  5       1    Opponent floor penalty / 14
  6       1    I hold first-player token (0 or 1)
  7       1    Opponent holds first-player token (0 or 1)
  8       5    My row completion per row r: filled_weighted / ((r+1)*5)
 13       5    Opponent row completion
 18       5    My col completion per col c: filled_weighted / 15
 23       5    Opponent col completion
 28       5    My color completion per color: filled_weighted / 15
 33       5    Opponent color completion
 38       5    Tiles available by color across all sources / 20
 43       5    Sources with at least one tile of that color / 5
 48       5    Bag tile count by color / 20

Row/col/color completion uses the post-placement wall (full pattern lines
treated as placed) so the network sees projected end-of-round wall state.

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
NUM_CHANNELS: int = 4
SPATIAL_SHAPE: tuple = (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)  # (4, 5, 5)

CH_MY_WALL: int = 0
CH_MY_PATTERN: int = 1
CH_OPP_WALL: int = 2
CH_OPP_PATTERN: int = 3

# ── Flat constants ─────────────────────────────────────────────────────────

OFF_MY_SCORE: int = 0
OFF_OPP_SCORE: int = 1
OFF_MY_EARNED: int = 2
OFF_OPP_EARNED: int = 3
OFF_MY_FLOOR: int = 4
OFF_OPP_FLOOR: int = 5
OFF_MY_FP_TOKEN: int = 6
OFF_OPP_FP_TOKEN: int = 7
OFF_MY_ROW_COMPLETION: int = 8  # 5 values, rows 0–4
OFF_OPP_ROW_COMPLETION: int = 13  # 5 values
OFF_MY_COL_COMPLETION: int = 18  # 5 values, cols 0–4
OFF_OPP_COL_COMPLETION: int = 23  # 5 values
OFF_MY_COLOR_COMPLETION: int = 28  # 5 values, COLOR_TILES order
OFF_OPP_COLOR_COMPLETION: int = 33  # 5 values
OFF_TILES_AVAILABLE: int = 38  # 5 values, COLOR_TILES order
OFF_SOURCES_WITH_COLOR: int = 43  # 5 values
OFF_BAG_COUNT: int = 48  # 5 values
FLAT_SIZE: int = 53

# ── Normalization constants ────────────────────────────────────────────────

MAX_SCORE_DIVISOR: float = 100.0
EARNED_DIVISOR: float = 100.0
FLOOR_PENALTY_DIVISOR: float = 14.0
MAX_BAG_TILES: float = float(TILES_PER_COLOR)  # 20
MAX_SOURCES: float = 5.0
_MAX_WEIGHTED_COL_COLOR: float = 15.0  # sum(1+2+3+4+5)

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


# ── Internal helpers — flat encoders ──────────────────────────────────────


def _encode_flat_scores(
    flat: torch.Tensor,
    my_player: Player,
    opp_player: Player,
) -> None:
    """Write official scores and earned values."""
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


def _encode_flat_row_completion(
    flat: torch.Tensor,
    offset: int,
    wall: list[list[Tile | None]],
) -> None:
    """Write row completion fraction for each row using post-placement wall."""
    filled = _count_filled_in_rows(wall)
    for row in range(BOARD_SIZE):
        flat[offset + row] = filled[row] / ((row + 1) * BOARD_SIZE)


def _encode_flat_col_completion(
    flat: torch.Tensor,
    offset: int,
    wall: list[list[Tile | None]],
) -> None:
    """Write column completion fraction for each column using post-placement wall."""
    filled = _count_filled_in_cols(wall)
    for col in range(BOARD_SIZE):
        flat[offset + col] = filled[col] / _MAX_WEIGHTED_COL_COLOR


def _encode_flat_color_completion(
    flat: torch.Tensor,
    offset: int,
    wall: list[list[Tile | None]],
) -> None:
    """Write color completion fraction for each color using post-placement wall."""
    filled = _count_filled_per_color(wall)
    for color_idx, color in enumerate(COLOR_TILES):
        flat[offset + color_idx] = filled[color] / _MAX_WEIGHTED_COL_COLOR


def _encode_flat_game_tiles(flat: torch.Tensor, game: Game) -> None:
    """Write tiles-available, sources-with-color, and bag-count features."""
    for color_idx, color in enumerate(COLOR_TILES):
        tiles_available = 0
        sources_with_color = 0
        for factory in game.factories:
            count = factory.count(color)
            if count > 0:
                tiles_available += count
                sources_with_color += 1
        center_count = game.center.count(color)
        if center_count > 0:
            tiles_available += center_count
            sources_with_color += 1
        flat[OFF_TILES_AVAILABLE + color_idx] = tiles_available / MAX_BAG_TILES
        flat[OFF_SOURCES_WITH_COLOR + color_idx] = sources_with_color / MAX_SOURCES
        flat[OFF_BAG_COUNT + color_idx] = game.bag.count(color) / MAX_BAG_TILES


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

    spatial : float32 tensor of shape SPATIAL_SHAPE = (4, 5, 5)
    flat    : float32 tensor of shape (FLAT_SIZE,) = (53,)
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
    _encode_wall_filled(spatial, CH_OPP_WALL, opp_player.wall)
    _encode_pattern_line_fill_ratio(
        spatial, CH_OPP_PATTERN, opp_player, opp_player.wall
    )

    flat = torch.zeros(FLAT_SIZE, dtype=torch.float32)

    _encode_flat_scores(flat, my_player, opp_player)
    _encode_flat_floor_penalties(flat, my_player, opp_player)
    _encode_flat_first_player_tokens(flat, my_player, opp_player)
    _encode_flat_row_completion(flat, OFF_MY_ROW_COMPLETION, my_post_wall)
    _encode_flat_row_completion(flat, OFF_OPP_ROW_COMPLETION, opp_post_wall)
    _encode_flat_col_completion(flat, OFF_MY_COL_COMPLETION, my_post_wall)
    _encode_flat_col_completion(flat, OFF_OPP_COL_COMPLETION, opp_post_wall)
    _encode_flat_color_completion(flat, OFF_MY_COLOR_COMPLETION, my_post_wall)
    _encode_flat_color_completion(flat, OFF_OPP_COLOR_COMPLETION, opp_post_wall)
    _encode_flat_game_tiles(flat, game)

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
