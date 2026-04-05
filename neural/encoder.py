# neural/encoder.py

"""Encodes Azul game states and moves as tensors / indices for the neural network.

State vector layout  (STATE_SIZE = 157 floats)
----------------------------------------------
All values normalized to [0.0, 1.0] unless noted.
Encoded from the *current player's* perspective — "my" always means the
player whose turn it is; "opp" always means the other player.

Offset  Size  Section
------  ----  -------
  0      25   My wall          — 1.0 if tile present, row-major (row*5+col)
 25      25   Opponent wall    — same
 50       5   My pattern line fill ratios     — tiles_present / row_capacity
 55      25   My pattern line colors          — one-hot per row (5 floats each)
 80       5   Opponent pattern line fill ratios
 85      25   Opponent pattern line colors    — one-hot per row (5 floats each)
110      25   Factories        — count of each color per factory / 4,
                                 laid out as factory_idx * 5 + color_idx
135       5   Center color counts             — count / TILES_PER_COLOR
140       1   First-player token in center    — 1.0 / 0.0
141       1   I hold the first-player token   — 1.0 / 0.0
142       1   My floor         — tiles on floor / 7
143       1   Opponent floor   — same
144       1   My earned score  — earned_score / 100
145       1   Opponent earned score — same
146       5   Bag totals       — count of each color / TILES_PER_COLOR
151       5   Discard totals   — count of each color / TILES_PER_COLOR
156       1   Score delta      — (my_earned - opp_earned) / 20, clamped [-1, 1]
------  ----
Total: 157

Move index layout
-----------------
Each move is a triple (source_idx, color_idx, dest_idx):
  source_idx  : 0 … NUM_FACTORIES-1 = factory;  NUM_FACTORIES = center
  color_idx   : 0 … BOARD_SIZE-1  (COLORS order)
  dest_idx    : 0 … BOARD_SIZE-1 = pattern line;  BOARD_SIZE = floor

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

# ── State-vector offsets ───────────────────────────────────────────────────

OFF_MY_WALL: int = 0
OFF_OPP_WALL: int = 25
OFF_MY_PL_FILL: int = 50
OFF_MY_PL_COLOR: int = 55
OFF_OPP_PL_FILL: int = 80
OFF_OPP_PL_COLOR: int = 85
OFF_FACTORIES: int = 110
OFF_CENTER: int = 135
OFF_FP_CENTER: int = 140
OFF_FP_MINE: int = 141
OFF_MY_FLOOR: int = 142
OFF_OPP_FLOOR: int = 143
OFF_MY_SCORE: int = 144
OFF_OPP_SCORE: int = 145
OFF_BAG: int = 146
OFF_DISCARD: int = 151
OFF_SCORE_DELTA: int = 156

STATE_SIZE: int = 157


# ── Internal helpers ───────────────────────────────────────────────────────


def _source_to_idx(source: int) -> int:
    """Convert a move source (factory index or CENTER sentinel) to 0-based int."""
    return NUM_FACTORIES if source == CENTER else source


def _idx_to_source(idx: int) -> int:
    """Convert a 0-based source index back to a factory index or CENTER."""
    return CENTER if idx == NUM_FACTORIES else idx


def _dest_to_idx(destination: int) -> int:
    """Convert a destination (pattern line index or FLOOR sentinel) to 0-based int."""
    return BOARD_SIZE if destination == FLOOR else destination


def _idx_to_dest(idx: int) -> int:
    """Convert a 0-based destination index back to a pattern line index or FLOOR."""
    return FLOOR if idx == BOARD_SIZE else idx


# ── Public API ─────────────────────────────────────────────────────────────


def encode_state(game: Game) -> torch.Tensor:
    """Encode a Game into a float32 vector of shape (STATE_SIZE,)."""
    v = torch.zeros(STATE_SIZE, dtype=torch.float32)
    cur = game.state.current_player
    opp = 1 - cur
    my = game.state.players[cur]
    op = game.state.players[opp]

    # My wall (0-24) and opponent wall (25-49)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if my.wall[row][col] is not None:
                v[OFF_MY_WALL + row * BOARD_SIZE + col] = 1.0
            if op.wall[row][col] is not None:
                v[OFF_OPP_WALL + row * BOARD_SIZE + col] = 1.0

    # My pattern lines (fill ratio + one-hot color)
    for row in range(BOARD_SIZE):
        line = my.pattern_lines[row]
        capacity = row + 1
        if line:
            v[OFF_MY_PL_FILL + row] = len(line) / capacity
            color_idx = COLOR_TILES.index(line[0])
            v[OFF_MY_PL_COLOR + row * BOARD_SIZE + color_idx] = 1.0

    # Opponent pattern lines (fill ratio + one-hot color)
    for row in range(BOARD_SIZE):
        line = op.pattern_lines[row]
        capacity = row + 1
        if line:
            v[OFF_OPP_PL_FILL + row] = len(line) / capacity
            color_idx = COLOR_TILES.index(line[0])
            v[OFF_OPP_PL_COLOR + row * BOARD_SIZE + color_idx] = 1.0

    # Factories — count of each color per factory, normalized by TILES_PER_FACTORY
    for f_idx, factory in enumerate(game.state.factories):
        for color_idx, color in enumerate(COLOR_TILES):
            count = factory.count(color)
            v[OFF_FACTORIES + f_idx * BOARD_SIZE + color_idx] = (
                count / TILES_PER_FACTORY
            )

    # Center — count of each color, normalized by TILES_PER_COLOR
    for color_idx, color in enumerate(COLOR_TILES):
        v[OFF_CENTER + color_idx] = game.state.center.count(color) / TILES_PER_COLOR

    # First player token
    v[OFF_FP_CENTER] = 1.0 if Tile.FIRST_PLAYER in game.state.center else 0.0
    v[OFF_FP_MINE] = 1.0 if Tile.FIRST_PLAYER in my.floor_line else 0.0

    # Floor lines
    v[OFF_MY_FLOOR] = len(my.floor_line) / 7
    v[OFF_OPP_FLOOR] = len(op.floor_line) / 7

    # Earned scores
    my_earned = earned_score(my)
    opp_earned = earned_score(op)
    v[OFF_MY_SCORE] = my_earned / 100
    v[OFF_OPP_SCORE] = opp_earned / 100

    # Score delta — who is winning right now, from current player's perspective.
    v[OFF_SCORE_DELTA] = max(-1.0, min(1.0, (my_earned - opp_earned) / 20))

    # Bag and discard totals
    for color_idx, color in enumerate(COLOR_TILES):
        v[OFF_BAG + color_idx] = game.state.bag.count(color) / TILES_PER_COLOR
        v[OFF_DISCARD + color_idx] = game.state.discard.count(color) / TILES_PER_COLOR

    return v


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
