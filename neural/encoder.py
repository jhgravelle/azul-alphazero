# neural/encoder.py

"""Encodes Azul game states and moves as tensors for the neural network.

Encoding v3: Flat vector (125 values)
---------------------------------------------------------------------
Single flat vector with board state and game state combined.

Wall & Pattern State (100 values):
  0–24     My pattern line fills (flattened 5×5 grid, 0.0–1.0 ratio)
  25–49      My wall (flattened 5×5 grid, binary)
  50–74     Opponent pattern line fills (flattened 5×5 grid, 0.0–1.0 ratio)
  75–99     Opponent wall (flattened 5×5 grid, binary)

Game State (25 values, laid out as a 5×5 grid):
  100       My official score / 100
  101       My earned score / 100
  102       0 (padding)
  103       Opponent official score / 100
  104       Opponent earned score / 100
  105       My floor penalty / 14
  106       I hold first-player token (0 or 1)
  107       0 (padding)
  108       Opponent floor penalty / 14
  109       Opponent holds first-player token (0 or 1)
  110–114   Tiles available by color (5 colors) / 20
  115–119   Sources with that color (5 colors) / 5
  120–124   Bag tile count by color (5 colors) / 20

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
import torch.nn.functional as F

from engine.constants import (
    SIZE,
    COLOR_TILES,
    COL_FOR_TILE_ROW,
    CUMULATIVE_FLOOR_PENALTIES,
    PLAYERS,
    TILES_PER_COLOR,
    Tile,
)
from engine.game import Game, Move, CENTER, FLOOR
from engine.player import Player

# ── Move-space constants ───────────────────────────────────────────────────

NUM_FACTORIES: int = 2 * PLAYERS + 1
NUM_SOURCES: int = NUM_FACTORIES + 1  # factories + center
NUM_DESTINATIONS: int = SIZE + 1  # pattern lines + floor
MOVE_SPACE_SIZE: int = NUM_SOURCES * SIZE * NUM_DESTINATIONS

# ── Encoding constants ────────────────────────────────────────────────────

NUM_COLORS: int = len(COLOR_TILES)  # 5
BOARD_CELLS: int = SIZE * SIZE  # 25

OFF_MY_PATTERN: int = 0  # 25 values
OFF_MY_WALL: int = 25  # 25 values
OFF_OPP_PATTERN: int = 50  # 25 values
OFF_OPP_WALL: int = 75  # 25 values

OFF_MY_SCORE: int = 100
OFF_MY_EARNED: int = 101
# index 102: padding zero
OFF_OPP_SCORE: int = 103
OFF_OPP_EARNED: int = 104
OFF_MY_FLOOR: int = 105
OFF_MY_FP_TOKEN: int = 106
# index 107: padding zero
OFF_OPP_FLOOR: int = 108
OFF_OPP_FP_TOKEN: int = 109
OFF_TILES_AVAILABLE: int = 110  # 5 values, COLOR_TILES order
OFF_SOURCES_WITH_COLOR: int = 115  # 5 values
OFF_BAG_COUNT: int = 120  # 5 values

FLAT_SIZE: int = 125

# ── Normalization constants ────────────────────────────────────────────────

MAX_SCORE_DIVISOR: float = 100.0
EARNED_DIVISOR: float = 100.0
FLOOR_PENALTY_DIVISOR: float = 14.0
MAX_BAG_TILES: float = float(TILES_PER_COLOR)  # 20
MAX_SOURCES: float = 5.0

# ── Internal helpers — board state encoders ───────────────────────────────


def _encode_wall_flattened(
    encoding: torch.Tensor,
    offset: int,
    wall: list[list[int]],
) -> None:
    """Write 1.0 for each occupied wall cell into flattened positions."""
    for row in range(SIZE):
        for wall_col in range(SIZE):
            if wall[row][wall_col]:
                flat_idx = row * SIZE + wall_col
                encoding[offset + flat_idx] = 1.0


def _encode_pattern_line_fill_ratio_flattened(
    encoding: torch.Tensor,
    offset: int,
    player: Player,
    wall: list[list[int]],
) -> None:
    """Write pattern line fill ratio into flattened positions."""
    for row in range(SIZE):
        tile = player._line_tile(row)
        if tile is None:
            continue
        wall_col = COL_FOR_TILE_ROW[tile][row]
        if wall[row][wall_col]:
            continue
        capacity = row + 1
        fill_count = player.pattern_grid[row][wall_col]
        if fill_count == 0:
            continue
        fill_ratio = fill_count / capacity
        flat_idx = row * SIZE + wall_col
        encoding[offset + flat_idx] = fill_ratio


# ── Internal helpers — flat encoders ──────────────────────────────────────


def _encode_flat_scores(
    flat: torch.Tensor,
    my_player: Player,
    opp_player: Player,
) -> None:
    """Write official scores and earned values."""
    flat[OFF_MY_SCORE] = my_player.score / MAX_SCORE_DIVISOR
    flat[OFF_MY_EARNED] = my_player.earned / EARNED_DIVISOR
    flat[OFF_OPP_SCORE] = opp_player.score / MAX_SCORE_DIVISOR
    flat[OFF_OPP_EARNED] = opp_player.earned / EARNED_DIVISOR


def _encode_flat_floor_penalties(
    flat: torch.Tensor,
    my_player: Player,
    opp_player: Player,
) -> None:
    """Write floor penalty values normalized by max penalty."""
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


def _encode_flat_game_tiles(flat: torch.Tensor, game: Game) -> None:
    """Write tiles-available, sources-with-color, and bag-count features."""
    availability = game.tile_availability()
    for color_idx, color in enumerate(COLOR_TILES):
        total_tiles, source_count = availability[color]
        flat[OFF_TILES_AVAILABLE + color_idx] = total_tiles / MAX_BAG_TILES
        flat[OFF_SOURCES_WITH_COLOR + color_idx] = source_count / MAX_SOURCES
        flat[OFF_BAG_COUNT + color_idx] = game.bag.count(color) / MAX_BAG_TILES


# ── Move index helpers ─────────────────────────────────────────────────────


def _source_to_idx(source: int) -> int:
    return NUM_FACTORIES if source == CENTER else source


def _idx_to_source(idx: int) -> int:
    return CENTER if idx == NUM_FACTORIES else idx


def _dest_to_idx(destination: int) -> int:
    return SIZE if destination == FLOOR else destination


def _idx_to_dest(idx: int) -> int:
    return FLOOR if idx == SIZE else idx


# ── Public API ─────────────────────────────────────────────────────────────


def encode_state(game: Game) -> torch.Tensor:
    """Encode a Game into a flat vector of 125 values.

    Returns float32 tensor of shape (125,) containing:
      0–24:  My pattern line fills (flattened)
      25–49:   My wall (flattened)
      50–74:  Opponent pattern line fills (flattened)
      75–99:  Opponent wall (flattened)
      100–124: Game state (scores, penalties, tokens, tiles, bag)
    """
    current_player_index = game.current_player_index
    opponent_index = 1 - current_player_index
    my_player = game.players[current_player_index]
    opp_player = game.players[opponent_index]

    encoding = torch.zeros(FLAT_SIZE, dtype=torch.float32)

    _encode_wall_flattened(encoding, OFF_MY_WALL, my_player.wall)
    _encode_wall_flattened(encoding, OFF_OPP_WALL, opp_player.wall)
    _encode_pattern_line_fill_ratio_flattened(
        encoding, OFF_MY_PATTERN, my_player, my_player.wall
    )
    _encode_pattern_line_fill_ratio_flattened(
        encoding, OFF_OPP_PATTERN, opp_player, opp_player.wall
    )

    _encode_flat_scores(encoding, my_player, opp_player)
    _encode_flat_floor_penalties(encoding, my_player, opp_player)
    _encode_flat_first_player_tokens(encoding, my_player, opp_player)
    _encode_flat_game_tiles(encoding, game)

    return encoding


def encode_move(move: Move, _game: Game) -> int:
    """Encode a Move as a unique integer index in [0, MOVE_SPACE_SIZE)."""
    source_idx = _source_to_idx(move.source)
    color_idx = COLOR_TILES.index(move.tile)
    dest_idx = _dest_to_idx(move.destination)
    return (
        source_idx * (SIZE * NUM_DESTINATIONS) + color_idx * NUM_DESTINATIONS + dest_idx
    )


def decode_move(index: int, _game: Game) -> Move:
    """Decode an integer index back into a Move."""
    source_idx, remainder = divmod(index, SIZE * NUM_DESTINATIONS)
    color_idx, dest_idx = divmod(remainder, NUM_DESTINATIONS)
    return Move(
        source=_idx_to_source(source_idx),
        tile=COLOR_TILES[color_idx],
        destination=_idx_to_dest(dest_idx),
    )


def priors_from_3head(
    src_logits: torch.Tensor,  # (2,)
    tile_logits: torch.Tensor,  # (5,)
    dst_logits: torch.Tensor,  # (6,)
    legal: "list[Move]",
    _game: "Game",
) -> list[float]:
    """Compute normalized per-move priors from 3-head policy logits.

    Prior for a move = softmax(src)[src_type] × softmax(tile)[t] × softmax(dst)[d],
    renormalized over legal moves so priors sum to 1.

    src_type: 0 = center, 1 = factory
    """
    src_probs = F.softmax(src_logits, dim=0)
    tile_probs = F.softmax(tile_logits, dim=0)
    dst_probs = F.softmax(dst_logits, dim=0)

    raw: list[float] = []
    for move in legal:
        src_type = 0 if move.source == CENTER else 1
        t = COLOR_TILES.index(move.tile)
        d = _dest_to_idx(move.destination)
        raw.append(float(src_probs[src_type] * tile_probs[t] * dst_probs[d]))

    total = sum(raw) or 1.0
    return [p / total for p in raw]


def flat_policy_to_3head_targets(
    flat_policy: torch.Tensor,  # (B, MOVE_SPACE_SIZE)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Marginalize flat MCTS visit distribution to 3-head training targets.

    Returns:
        src_targets:  (B, 2) — center vs factory marginal
        tile_targets: (B, 5) — color marginal
        dst_targets:  (B, 6) — destination marginal
    """
    B = flat_policy.shape[0]
    # Reshape to (B, NUM_SOURCES=7, BOARD_SIZE=5, NUM_DESTINATIONS=6)
    p3 = flat_policy.view(B, NUM_SOURCES, SIZE, NUM_DESTINATIONS)

    tile_targets = p3.sum(dim=1).sum(dim=2)  # (B, 5)
    dst_targets = p3.sum(dim=1).sum(dim=1)  # (B, 6)

    # Source type: index NUM_FACTORIES (=6) is center, 0..5 are factories
    center_mass = p3[:, NUM_FACTORIES, :, :].sum(dim=(1, 2))  # (B,)
    factory_mass = p3[:, :NUM_FACTORIES, :, :].sum(dim=(1, 2, 3))  # (B,)
    src_targets = torch.stack([center_mass, factory_mass], dim=1)  # (B, 2)

    return src_targets, tile_targets, dst_targets


def format_encoding(encoding: torch.Tensor) -> str:
    """Format a 125-value encoding tensor as five 5x5 grids printed side by side.

    Grids left to right: my wall, opponent wall, my pattern, opponent pattern,
    game state. Each cell is right-aligned to 5 characters (e.g. ' 0.00' or
    ' 1.00') with two spaces between columns inside a grid and four spaces
    between grids so columns stay visually aligned.
    """
    grids = [
        encoding[offset : offset + 25].reshape(5, 5) for offset in (0, 25, 50, 75, 100)
    ]
    lines = []
    for row_index in range(5):
        grid_strings = [
            "  ".join(
                f"{grids[grid_index][row_index, col].item():5.2f}" for col in range(5)
            )
            for grid_index in range(5)
        ]
        lines.append("    ".join(grid_strings))
    return "\n".join(lines)


if __name__ == "__main__":
    game = Game()
    game.setup_round()
    encoding = encode_state(game)
    print(format_encoding(encoding))
