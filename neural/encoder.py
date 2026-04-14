# neural/encoder.py

"""Encodes Azul game states and moves as tensors for the neural network.

Spatial tensor: shape (12, 5, 6)  — wall + pattern lines for both players
---------------------------------------------------------------------
Channels 0–5:  current player's planes
Channels 6–11: opponent's planes

Within each group of 6 channels:
  Channels 0–4: one plane per color (COLOR_TILES order)
  Channel 5:    "any tile" plane — 1.0 wherever any tile is present

Within each plane:
  Rows 0–4:    wall rows (= pattern line rows)
  Cols 0–4:    wall columns — 1.0 if that cell is filled
  Col 5:       pattern line fill ratio for this row, in the color's plane
               (0.0 if the line is empty or a different color)
               in the "any" plane, col 5 = fill ratio regardless of color

Flat vector: shape (FLAT_SIZE,)
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
 34       1   My earned score / 100
 35       1   Opponent earned score / 100
 36       1   Score delta — (my_earned - opp_earned) / 20, clamped [-1, 1]
 37       5   Bag totals — count of each color / TILES_PER_COLOR
 42       5   Discard totals — count of each color / TILES_PER_COLOR
------  ----
Total: 47

Move index layout  (unchanged from previous encoder)
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
)
from engine.scoring import earned_score

# ── Move-space constants ───────────────────────────────────────────────────

NUM_FACTORIES: int = 2 * PLAYERS + 1
NUM_SOURCES: int = NUM_FACTORIES + 1  # factories + center
NUM_DESTINATIONS: int = BOARD_SIZE + 1  # pattern lines + floor
MOVE_SPACE_SIZE: int = NUM_SOURCES * BOARD_SIZE * NUM_DESTINATIONS

# ── Spatial constants ──────────────────────────────────────────────────────

NUM_COLORS: int = len(COLOR_TILES)  # 5
PLANES_PER_PLAYER: int = NUM_COLORS + 1  # 5 color planes + 1 "any" plane
NUM_CHANNELS: int = PLANES_PER_PLAYER * 2  # 12 — current player then opponent
WALL_COLS: int = BOARD_SIZE  # 5
PATTERN_COL: int = BOARD_SIZE  # column index 5
GRID_COLS: int = BOARD_SIZE + 1  # 6 — wall cols + pattern line col
SPATIAL_SHAPE: tuple = (NUM_CHANNELS, BOARD_SIZE, GRID_COLS)  # (12, 5, 6)

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
FLAT_SIZE: int = 47

# ── Internal helpers ───────────────────────────────────────────────────────


def _encode_player_planes(
    spatial: torch.Tensor,
    channel_offset: int,
    board,
) -> None:
    """Fill 6 channels of the spatial tensor for one player.

    channel_offset=0 for current player, 6 for opponent.
    Mutates spatial in-place.
    """
    any_channel = channel_offset + NUM_COLORS  # channel 5 or 11

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

    spatial: float32 tensor of shape SPATIAL_SHAPE = (12, 5, 6)
    flat:    float32 tensor of shape (FLAT_SIZE,) = (47,)
    """
    cur = game.state.current_player
    opp = 1 - cur
    my = game.state.players[cur]
    op = game.state.players[opp]

    # ── Spatial ───────────────────────────────────────────────────────────
    spatial = torch.zeros(SPATIAL_SHAPE, dtype=torch.float32)
    _encode_player_planes(spatial, channel_offset=0, board=my)
    _encode_player_planes(spatial, channel_offset=PLANES_PER_PLAYER, board=op)

    # ── Flat ──────────────────────────────────────────────────────────────
    flat = torch.zeros(FLAT_SIZE, dtype=torch.float32)

    # Factories
    for f_idx, factory in enumerate(game.state.factories):
        for color_idx, color in enumerate(COLOR_TILES):
            flat[OFF_FACTORIES + f_idx * NUM_COLORS + color_idx] = (
                factory.count(color) / TILES_PER_FACTORY
            )

    # Center
    for color_idx, color in enumerate(COLOR_TILES):
        flat[OFF_CENTER + color_idx] = game.state.center.count(color) / TILES_PER_COLOR

    # First player token
    flat[OFF_FP_CENTER] = 1.0 if Tile.FIRST_PLAYER in game.state.center else 0.0
    flat[OFF_FP_MINE] = 1.0 if Tile.FIRST_PLAYER in my.floor_line else 0.0

    # Floor lines
    flat[OFF_MY_FLOOR] = len(my.floor_line) / 7
    flat[OFF_OPP_FLOOR] = len(op.floor_line) / 7

    # Earned scores
    my_earned = earned_score(my)
    opp_earned = earned_score(op)
    flat[OFF_MY_SCORE] = my_earned / 100
    flat[OFF_OPP_SCORE] = opp_earned / 100

    # Score delta
    flat[OFF_SCORE_DELTA] = max(-1.0, min(1.0, (my_earned - opp_earned) / 20))

    # Bag and discard
    for color_idx, color in enumerate(COLOR_TILES):
        flat[OFF_BAG + color_idx] = game.state.bag.count(color) / TILES_PER_COLOR
        flat[OFF_DISCARD + color_idx] = (
            game.state.discard.count(color) / TILES_PER_COLOR
        )

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
