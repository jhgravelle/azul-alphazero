# neural/encoder.py

"""Encodes Azul game states and moves as tensors for the neural network.

Spatial tensor: shape (14, 5, 6)
---------------------------------------------------------------------
Channels 0–5:  current player's planes
Channel  6:    current player's blocked_wall plane
Channels 7–12: opponent's planes
Channel  13:   opponent's blocked_wall plane

Within the 6 color+any planes per player (channels 0–5 and 7–12):
  Channels 0–4: one plane per color (COLOR_TILES order)
  Channel 5:    "any tile" plane — 1.0 wherever any tile is present

Within each color or any plane:
  Rows 0–4:    wall rows (= pattern line rows)
  Cols 0–4:    wall columns — 1.0 if that cell is filled
  Col 5:       pattern line fill ratio for this row, in the color's plane
               (0.0 if the line is empty or a different color)
               in the "any" plane, col 5 = fill ratio regardless of color

blocked_wall plane (channel 6 / 13):
  Rows 0–4, Cols 0–4 only (col 5 is always 0.0).
  A cell [row, col] is 1.0 if it is blocked, meaning either:
    - The wall cell is already filled (tile placed), OR
    - The pattern line for that row is committed to a color whose wall
      column is NOT col — that cell cannot be filled until the pattern
      line is cleared and a new color is chosen.
  A cell is 0.0 (open) only when it is empty AND either the pattern line
  for that row is empty or the pattern line is committed to the color that
  belongs in that column.

Flat vector: shape (FLAT_SIZE,) = (49,)
---------------------------------------------------------------------
Offset  Size  Section
------  ----  -------
  0      25   Factories — count of each color per factory / TILES_PER_FACTORY
               layout: factory_idx * 5 + color_idx
 25       5   Center color counts — count / TILES_PER_COLOR
 30       1   First-player token in center — 1.0 / 0.0
 31       1   I hold the first-player token — 1.0 / 0.0
 32       1   My floor fill ratio — tiles on floor / 7
 33       1   Opponent floor fill ratio
 34       1   My earned_score_unclamped / 100
 35       1   Opponent earned_score_unclamped / 100
 36       1   Score delta — (my_unclamped - opp_unclamped) / 50, clamped [-1, 1]
 37       5   Bag totals — count of each color / TILES_PER_COLOR
 42       5   Discard totals — count of each color / TILES_PER_COLOR
 47       1   Round progress — (round - 1) / 5  (0.0 at round 1, 1.0 at round 6)
 48       1   Distinct source-color pairs / 10  (round countdown proxy)
------  ----
Total: 49

Move index layout  (unchanged)
-----------------------------------------------------
Each move is a triple (source_idx, color_idx, dest_idx):
  source_idx : 0 … NUM_FACTORIES-1 = factory;  NUM_FACTORIES = center
  color_idx  : 0 … BOARD_SIZE-1  (COLOR_TILES order)
  dest_idx   : 0 … BOARD_SIZE-1 = pattern line;  BOARD_SIZE = floor

Flat index = source_idx * (BOARD_SIZE * NUM_DESTINATIONS)
           + color_idx  * NUM_DESTINATIONS
           + dest_idx

MOVE_SPACE_SIZE = NUM_SOURCES * BOARD_SIZE * NUM_DESTINATIONS
"""

import torch
from engine.game import Game, Move, CENTER, FLOOR
from engine.constants import (
    Tile,
    BOARD_SIZE,
    COLOR_TILES,
    PLAYERS,
    TILES_PER_COLOR,
    TILES_PER_FACTORY,
    COLUMN_FOR_TILE_IN_ROW,
)
from engine.scoring import earned_score_unclamped

# ── Move-space constants ───────────────────────────────────────────────────

NUM_FACTORIES: int = 2 * PLAYERS + 1
NUM_SOURCES: int = NUM_FACTORIES + 1  # factories + center
NUM_DESTINATIONS: int = BOARD_SIZE + 1  # pattern lines + floor
MOVE_SPACE_SIZE: int = NUM_SOURCES * BOARD_SIZE * NUM_DESTINATIONS

# ── Spatial constants ──────────────────────────────────────────────────────

NUM_COLORS: int = len(COLOR_TILES)  # 5
PLANES_PER_PLAYER: int = NUM_COLORS + 1  # 5 color planes + 1 "any" plane
BLOCKED_WALL_CHANNEL_MY: int = PLANES_PER_PLAYER  # channel 6
BLOCKED_WALL_CHANNEL_OPP: int = PLANES_PER_PLAYER * 2 + 1  # channel 13
NUM_CHANNELS: int = (PLANES_PER_PLAYER + 1) * 2  # 14 — (6 + 1) per player × 2
WALL_COLS: int = BOARD_SIZE  # 5
PATTERN_COL: int = BOARD_SIZE  # column index 5
GRID_COLS: int = BOARD_SIZE + 1  # 6 — wall cols + pattern line col
SPATIAL_SHAPE: tuple = (NUM_CHANNELS, BOARD_SIZE, GRID_COLS)  # (14, 5, 6)

# ── Flat vector offsets ────────────────────────────────────────────────────

OFF_FACTORIES: int = 0
OFF_CENTER: int = 25
OFF_FP_CENTER: int = 30
OFF_FP_MINE: int = 31
OFF_MY_FLOOR: int = 32
OFF_OPP_FLOOR: int = 33
OFF_MY_SCORE: int = 34
OFF_OPP_SCORE: int = 35
OFF_SCORE_DELTA: int = 36
OFF_BAG: int = 37
OFF_DISCARD: int = 42
OFF_ROUND: int = 47
OFF_DISTINCT_PAIRS: int = 48
FLAT_SIZE: int = 49

# ── Score delta normalization ──────────────────────────────────────────────

SCORE_DELTA_DIVISOR: float = 50.0

# ── Internal helpers ───────────────────────────────────────────────────────


def _encode_player_planes(
    spatial: torch.Tensor,
    channel_offset: int,
    board,
) -> None:
    """Fill 6 color+any channels of the spatial tensor for one player.

    channel_offset=0 for current player, 7 for opponent (leaves room for the
    blocked_wall channel at offset 6 / 13).
    Mutates spatial in-place.
    """
    any_channel = channel_offset + NUM_COLORS  # channel 5 or 12

    for row in range(BOARD_SIZE):
        # Wall columns 0–4
        for col in range(BOARD_SIZE):
            tile = board.wall[row][col]
            if tile is not None:
                color_idx = COLOR_TILES.index(tile)
                spatial[channel_offset + color_idx, row, col] = 1.0
                spatial[any_channel, row, col] = 1.0

        # Pattern line — column 5
        line = board.pattern_lines[row]
        if line:
            capacity = row + 1
            fill_ratio = len(line) / capacity
            color_idx = COLOR_TILES.index(line[0])
            spatial[channel_offset + color_idx, row, PATTERN_COL] = fill_ratio
            spatial[any_channel, row, PATTERN_COL] = fill_ratio


def _encode_blocked_wall_plane(
    spatial: torch.Tensor,
    blocked_channel: int,
    board,
) -> None:
    """Fill the blocked_wall channel for one player.

    A wall cell [row, col] is blocked (1.0) if either:
      - The wall cell is already filled, OR
      - The pattern line for that row is committed to a color whose correct
        wall column is not col (so col can never be filled this round).

    Only wall columns 0–4 are written; col 5 (pattern column) is always 0.0.
    Mutates spatial in-place.
    """
    for row in range(BOARD_SIZE):
        committed_color = (
            board.pattern_lines[row][0] if board.pattern_lines[row] else None
        )
        committed_col = (
            COLUMN_FOR_TILE_IN_ROW[committed_color][row]
            if committed_color is not None
            else None
        )

        for col in range(BOARD_SIZE):
            wall_cell_is_filled = board.wall[row][col] is not None
            pattern_line_blocks_this_col = (
                committed_col is not None and col != committed_col
            )
            if wall_cell_is_filled or pattern_line_blocks_this_col:
                spatial[blocked_channel, row, col] = 1.0


def _source_to_idx(source: int) -> int:
    return NUM_FACTORIES if source == CENTER else source


def _idx_to_source(idx: int) -> int:
    return CENTER if idx == NUM_FACTORIES else idx


def _dest_to_idx(destination: int) -> int:
    return BOARD_SIZE if destination == FLOOR else destination


def _idx_to_dest(idx: int) -> int:
    return FLOOR if idx == BOARD_SIZE else idx


def _encode_flat_factories(flat: torch.Tensor, game: Game) -> None:
    """Write factory tile counts into the flat vector."""
    for factory_idx, factory in enumerate(game.state.factories):
        for color_idx, color in enumerate(COLOR_TILES):
            flat[OFF_FACTORIES + factory_idx * NUM_COLORS + color_idx] = (
                factory.count(color) / TILES_PER_FACTORY
            )


def _encode_flat_center(flat: torch.Tensor, game: Game) -> None:
    """Write center tile counts and first-player token flags."""
    for color_idx, color in enumerate(COLOR_TILES):
        flat[OFF_CENTER + color_idx] = game.state.center.count(color) / TILES_PER_COLOR
    flat[OFF_FP_CENTER] = 1.0 if Tile.FIRST_PLAYER in game.state.center else 0.0


def _encode_flat_floors(
    flat: torch.Tensor,
    my_board,
    opp_board,
) -> None:
    """Write floor fill ratios and first-player token on my floor."""
    flat[OFF_FP_MINE] = 1.0 if Tile.FIRST_PLAYER in my_board.floor_line else 0.0
    flat[OFF_MY_FLOOR] = len(my_board.floor_line) / 7
    flat[OFF_OPP_FLOOR] = len(opp_board.floor_line) / 7


def _encode_flat_scores(
    flat: torch.Tensor,
    my_board,
    opp_board,
) -> None:
    """Write earned_score_unclamped for both players and their delta."""
    my_unclamped = earned_score_unclamped(my_board)
    opp_unclamped = earned_score_unclamped(opp_board)
    flat[OFF_MY_SCORE] = my_unclamped / 100
    flat[OFF_OPP_SCORE] = opp_unclamped / 100
    raw_delta = (my_unclamped - opp_unclamped) / SCORE_DELTA_DIVISOR
    flat[OFF_SCORE_DELTA] = max(-1.0, min(1.0, raw_delta))


def _encode_flat_bag_and_discard(flat: torch.Tensor, game: Game) -> None:
    """Write bag and discard tile counts."""
    for color_idx, color in enumerate(COLOR_TILES):
        flat[OFF_BAG + color_idx] = game.state.bag.count(color) / TILES_PER_COLOR
        flat[OFF_DISCARD + color_idx] = (
            game.state.discard.count(color) / TILES_PER_COLOR
        )


def _encode_flat_round_progress(flat: torch.Tensor, game: Game) -> None:
    """Write round progress: (round - 1) / 5, giving 0.0 at round 1, 1.0 at round 6."""
    flat[OFF_ROUND] = (game.state.round - 1) / 5


def _encode_flat_distinct_pairs(flat: torch.Tensor, game: Game) -> None:
    """Write the distinct source-color pair count / 10 as a round countdown."""
    flat[OFF_DISTINCT_PAIRS] = game.count_distinct_source_color_pairs() / 10


# ── Public API ─────────────────────────────────────────────────────────────


def encode_state(game: Game) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a Game into (spatial, flat) tensors.

    spatial: float32 tensor of shape SPATIAL_SHAPE = (14, 5, 6)
    flat:    float32 tensor of shape (FLAT_SIZE,) = (49,)
    """
    current_player = game.state.current_player
    opponent = 1 - current_player
    my_board = game.state.players[current_player]
    opp_board = game.state.players[opponent]

    # ── Spatial ───────────────────────────────────────────────────────────
    spatial = torch.zeros(SPATIAL_SHAPE, dtype=torch.float32)

    # Color+any planes: current player at channels 0–5, opponent at 7–12
    _encode_player_planes(spatial, channel_offset=0, board=my_board)
    _encode_player_planes(
        spatial, channel_offset=PLANES_PER_PLAYER + 1, board=opp_board
    )

    # Blocked wall planes: channel 6 for current player, 13 for opponent
    _encode_blocked_wall_plane(spatial, BLOCKED_WALL_CHANNEL_MY, my_board)
    _encode_blocked_wall_plane(spatial, BLOCKED_WALL_CHANNEL_OPP, opp_board)

    # ── Flat ──────────────────────────────────────────────────────────────
    flat = torch.zeros(FLAT_SIZE, dtype=torch.float32)

    _encode_flat_factories(flat, game)
    _encode_flat_center(flat, game)
    _encode_flat_floors(flat, my_board, opp_board)
    _encode_flat_scores(flat, my_board, opp_board)
    _encode_flat_bag_and_discard(flat, game)
    _encode_flat_round_progress(flat, game)
    _encode_flat_distinct_pairs(flat, game)

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
