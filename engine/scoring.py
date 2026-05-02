# engine/scoring.py

"""Pure scoring functions for Azul.

All functions here are side-effect-free — they read board/wall state and
return integers or plain dataclasses. Mutation (clearing floors, moving
tiles to discard, updating board.score) stays in Game.
"""

from dataclasses import dataclass
from typing import Literal

from engine.board import Board
from engine.constants import (
    BOARD_SIZE,
    BONUS_TILE,
    BONUS_COLUMN,
    BONUS_ROW,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
    CUMULATIVE_FLOOR_PENALTIES,
    Tile,
)

# ── Plain dataclasses (no Pydantic dependency) ─────────────────────────────────


@dataclass
class PlacementDetail:
    """Pending wall placement for a single full pattern line."""

    row: int
    column: int
    placement_points: int


@dataclass
class BonusDetail:
    """An end-of-game bonus already guaranteed by the current wall state."""

    bonus_type: Literal["row", "column", "tile"]
    index: int  # row index, column index, or Tile enum value
    bonus_points: int


# ── Primitive scoring ──────────────────────────────────────────────────────────


def score_placement(wall: list[list[Tile | None]], row: int, column: int) -> int:
    """Score a single tile at (row, column) on the wall.

    Precondition: wall[row][column] must already be set to the placed tile
    before this function is called. The caller owns this responsibility.

    Counts contiguous horizontal and vertical runs through (row, column),
    including the tile itself. A lone tile with no neighbours scores 1.
    """
    h_start, h_end = column, column
    while h_start - 1 >= 0 and wall[row][h_start - 1] is not None:
        h_start -= 1
    while h_end + 1 < BOARD_SIZE and wall[row][h_end + 1] is not None:
        h_end += 1
    h = h_end - h_start + 1

    v_start, v_end = row, row
    while v_start - 1 >= 0 and wall[v_start - 1][column] is not None:
        v_start -= 1
    while v_end + 1 < BOARD_SIZE and wall[v_end + 1][column] is not None:
        v_end += 1
    v = v_end - v_start + 1

    return (h if h > 1 else 0) + (v if v > 1 else 0) or 1


def score_floor_penalty(floor_line: list[Tile]) -> int:
    """Return the penalty for the current floor line as a negative integer (or 0).

    Every tile — including FIRST_PLAYER — occupies a penalty slot. Uses a
    precomputed cumulative table so this is a single index lookup.
    """
    return CUMULATIVE_FLOOR_PENALTIES[len(floor_line)]


def score_wall_bonus(wall: list[list[Tile | None]]) -> int:
    """Return end-of-game bonus points already guaranteed by the current wall.

    Awards +BONUS_ROW per completed row, +BONUS_COLUMN per completed column,
    +BONUS_TILE per color with all 5 tiles placed.
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
            total += BONUS_TILE
    return total


# ── Pending scoring breakdown ──────────────────────────────────────────────────


def pending_placement_details(
    board: Board,
) -> tuple[list[PlacementDetail], list[list[Tile | None]]]:
    """Return placement details for all full pattern lines and the resulting wall.

    Simulates end-of-round tile placement in row order on a temporary copy of
    the wall so that adjacency scores are correct when pending placements are
    adjacent to each other. Does not mutate board.wall.

    Returns a tuple of:
      - list of PlacementDetail, one per full pattern line
      - the temporary wall with all pending placements applied
    """
    wall: list[list[Tile | None]] = [row[:] for row in board.wall]
    details: list[PlacementDetail] = []

    for row, line in enumerate(board.pattern_lines):
        if len(line) < row + 1:
            continue
        tile = line[0]
        column = COLUMN_FOR_TILE_IN_ROW[tile][row]
        wall[row][column] = tile
        points = score_placement(wall, row, column)
        details.append(PlacementDetail(row=row, column=column, placement_points=points))

    return details, wall


def pending_bonus_details(wall: list[list[Tile | None]]) -> list[BonusDetail]:
    """Return bonus details for all end-of-game bonuses guaranteed by the wall.

    Takes a wall directly rather than a Board so it can be called with the
    post-simulation wall returned by pending_placement_details.
    """
    bonuses: list[BonusDetail] = []

    for row in range(BOARD_SIZE):
        if all(wall[row][column] is not None for column in range(BOARD_SIZE)):
            bonuses.append(
                BonusDetail(bonus_type="row", index=row, bonus_points=BONUS_ROW)
            )

    for column in range(BOARD_SIZE):
        if all(wall[row][column] is not None for row in range(BOARD_SIZE)):
            bonuses.append(
                BonusDetail(
                    bonus_type="column", index=column, bonus_points=BONUS_COLUMN
                )
            )

    for tile in COLOR_TILES:
        if all(
            wall[row][COLUMN_FOR_TILE_IN_ROW[tile][row]] == tile
            for row in range(BOARD_SIZE)
        ):
            bonuses.append(
                BonusDetail(
                    bonus_type="tile", index=tile.value, bonus_points=BONUS_TILE
                )
            )

    return bonuses


# ── Composite scoring ──────────────────────────────────────────────────────────


def earned_score(board: Board) -> int:
    """Return points earned this round but not yet applied to board.score.

    Includes placement points for full pattern lines, floor penalty, and
    any end-of-game bonuses guaranteed by the post-placement wall.
    Can be negative before clamping — result is clamped to 0.
    """
    details, temp_wall = pending_placement_details(board)
    total = (
        board.score
        + sum(d.placement_points for d in details)
        + score_floor_penalty(board.floor_line)
        + score_wall_bonus(temp_wall)
    )
    return max(0, total)


def earned_score_unclamped(board: Board) -> int:
    details, temp_wall = pending_placement_details(board)
    return (
        board.score
        + sum(d.placement_points for d in details)
        + score_floor_penalty(board.floor_line)
        + score_wall_bonus(temp_wall)
    )
