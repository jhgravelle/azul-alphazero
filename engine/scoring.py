# engine/scoring.py

"""Pure scoring functions for Azul.

All functions here are side-effect-free — they read board/wall state and
return integers. Mutation (clearing floors, moving tiles to discard, updating
board.score) stays in Game.
"""

from engine.board import Board
from engine.constants import (
    Tile,
    BOARD_SIZE,
    BONUS_COLOR,
    BONUS_COLUMN,
    BONUS_ROW,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
    CUMULATIVE_FLOOR_PENALTIES,
)


def score_placement(wall: list[list[Tile | None]], row: int, col: int) -> int:
    """Score a single tile at (row, col) on the wall.

    Precondition: wall[row][col] must already be set to the placed tile
    before this function is called. The caller owns this responsibility.

    Counts contiguous horizontal and vertical runs through (row, col),
    including the tile itself. A lone tile with no neighbours scores 1.
    """
    h_start, h_end = col, col
    while h_start - 1 >= 0 and wall[row][h_start - 1] is not None:
        h_start -= 1
    while h_end + 1 < BOARD_SIZE and wall[row][h_end + 1] is not None:
        h_end += 1
    h = h_end - h_start + 1

    v_start, v_end = row, row
    while v_start - 1 >= 0 and wall[v_start - 1][col] is not None:
        v_start -= 1
    while v_end + 1 < BOARD_SIZE and wall[v_end + 1][col] is not None:
        v_end += 1
    v = v_end - v_start + 1

    return (h if h > 1 else 0) + (v if v > 1 else 0) or 1


def score_floor_penalty(floor_line: list[Tile]) -> int:
    """Return the penalty for the current floor line as a negative integer (or 0).

    Counts up to len(FLOOR_PENALTIES) slots. Every tile — including
    FIRST_PLAYER — occupies a penalty slot. Uses a precomputed cumulative
    table (_FLOOR_CUM) so this is a single index lookup.
    """
    return CUMULATIVE_FLOOR_PENALTIES[len(floor_line)]


def score_wall_bonus(wall: list[list[Tile | None]]) -> int:
    """Return end-of-game bonus points already guaranteed by the current wall.

    Awards +BONUS_ROW per completed row, +BONUS_COLUMN per completed column,
    +BONUS_COLOR per color with all 5 tiles placed.
    """
    total = 0

    for row in range(BOARD_SIZE):
        if all(wall[row][column] is not None for column in range(BOARD_SIZE)):
            total += BONUS_ROW

    for column in range(BOARD_SIZE):
        if all(wall[row][column] is not None for row in range(BOARD_SIZE)):
            total += BONUS_COLUMN

    for tile in COLOR_TILES:
        if all(
            wall[row][COLUMN_FOR_TILE_IN_ROW[tile][row]] == tile
            for row in range(BOARD_SIZE)
        ):
            total += BONUS_COLOR

    return total


def earned_score(board: Board) -> int:
    """Return points earned this round but not yet applied to board.score.

    Includes three components:
    - Wall placement score for each currently full pattern line. Placements
      are simulated in row order on a temporary copy of the wall, so earlier
      placements are visible to later ones when their cells are adjacent.
    - Floor penalty for tiles currently on the floor line.
    - End-of-game bonuses for completed rows (+2), columns (+7), and colors
      (+10) already on the wall (via wall_bonus_score).

    Can be negative if floor penalties exceed placement gains.
    """
    total = board.score

    # Shallow-copy each row so we can place tiles without mutating board.wall.
    wall: list[list[Tile | None]] = [row[:] for row in board.wall]

    # --- Full pattern lines: simulate placements in row order ---
    for row, line in enumerate(board.pattern_lines):
        if len(line) < row + 1:
            continue
        tile = line[0]
        column = COLUMN_FOR_TILE_IN_ROW[tile][row]
        wall[row][column] = tile
        total += score_placement(wall, row, column)

    # --- Floor penalty ---
    total += score_floor_penalty(board.floor_line)

    # --- End-of-game bonuses (wall + pending placements) ---
    total += score_wall_bonus(wall)

    return total if total > 0 else 0
