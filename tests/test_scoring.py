# tests/test_scoring.py

"""Tests for scoring — pure functions in engine/scoring.py and Game scoring methods."""

from engine.constants import (
    BOARD_SIZE,
    BONUS_TILE,
    BONUS_COLUMN,
    BONUS_ROW,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
    FLOOR_PENALTIES,
    Tile,
    WALL_PATTERN,
)
from engine.board import Board
from engine.scoring import (
    BonusDetail,
    PlacementDetail,
    earned_score,
    pending_bonus_details,
    pending_placement_details,
    score_floor_penalty,
    score_placement,
    score_wall_bonus,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _empty_wall() -> list[list[Tile | None]]:
    return [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]


# ── score_placement ────────────────────────────────────────────────────────────


def test_score_placement_lone_tile_scores_one():
    wall = _empty_wall()
    wall[2][2] = Tile.BLUE
    assert score_placement(wall, 2, 2) == 1


def test_score_placement_two_horizontal_neighbors_left():
    wall = _empty_wall()
    wall[2][2] = Tile.BLUE
    wall[2][1] = Tile.WHITE
    assert score_placement(wall, 2, 1) == 2


def test_score_placement_two_horizontal_neighbors_right():
    wall = _empty_wall()
    wall[2][2] = Tile.BLUE
    wall[2][3] = Tile.YELLOW
    assert score_placement(wall, 2, 3) == 2


def test_score_placement_two_horizontal_neighbors_center():
    wall = _empty_wall()
    wall[2][1] = Tile.WHITE
    wall[2][3] = Tile.YELLOW
    wall[2][2] = Tile.BLUE
    assert score_placement(wall, 2, 2) == 3


def test_score_placement_two_horizontal_neighbors_with_gap():
    wall = _empty_wall()
    wall[2][0] = Tile.BLACK
    wall[2][2] = Tile.BLUE
    wall[2][3] = Tile.YELLOW
    assert score_placement(wall, 2, 3) == 2


def test_score_placement_two_vertical_neighbors_up():
    wall = _empty_wall()
    wall[2][2] = Tile.BLUE
    wall[1][2] = Tile.RED
    assert score_placement(wall, 1, 2) == 2


def test_score_placement_two_vertical_neighbors_down():
    wall = _empty_wall()
    wall[2][2] = Tile.BLUE
    wall[3][2] = Tile.WHITE
    assert score_placement(wall, 3, 2) == 2


def test_score_placement_two_vertical_neighbors_center():
    wall = _empty_wall()
    wall[1][2] = Tile.YELLOW
    wall[3][2] = Tile.WHITE
    wall[2][2] = Tile.BLUE
    assert score_placement(wall, 2, 2) == 3


def test_score_placement_two_vertical_neighbors_with_gap():
    wall = _empty_wall()
    wall[0][2] = Tile.RED
    wall[2][2] = Tile.BLUE
    wall[3][2] = Tile.WHITE
    assert score_placement(wall, 3, 2) == 2


def test_score_placement_both_directions_scores_sum():
    wall = _empty_wall()
    wall[1][2] = Tile.YELLOW
    wall[2][1] = Tile.WHITE
    wall[2][3] = Tile.YELLOW
    wall[3][2] = Tile.WHITE
    wall[2][2] = Tile.BLUE
    assert score_placement(wall, 2, 2) == 6


def test_score_placement_l_shaped_one_directions_scores_one():
    wall = _empty_wall()
    wall[1][2] = Tile.YELLOW
    wall[2][2] = Tile.BLUE
    wall[2][3] = Tile.YELLOW
    assert score_placement(wall, 2, 3) == 2


# ── score_floor_penalty ────────────────────────────────────────────────────────


def test_score_floor_penalty_empty_floor_is_zero():
    assert score_floor_penalty([]) == 0


def test_score_floor_penalty_one_tile():
    assert score_floor_penalty([Tile.BLUE]) == FLOOR_PENALTIES[0]


def test_score_floor_penalty_two_tiles():
    expected = FLOOR_PENALTIES[0] + FLOOR_PENALTIES[1]
    assert score_floor_penalty([Tile.BLUE, Tile.RED]) == expected


def test_score_floor_penalty_seven_tiles_uses_all_slots():
    assert score_floor_penalty([Tile.BLUE] * 7) == sum(FLOOR_PENALTIES)


def test_score_floor_penalty_more_than_seven_tiles_capped():
    assert score_floor_penalty([Tile.BLUE] * 10) == sum(FLOOR_PENALTIES)


def test_score_floor_penalty_first_player_marker_counts_as_a_slot():
    tiles = [Tile.FIRST_PLAYER, Tile.BLUE]
    expected = FLOOR_PENALTIES[0] + FLOOR_PENALTIES[1]
    assert score_floor_penalty(tiles) == expected


def test_score_floor_penalty_is_negative_or_zero():
    for n in range(8):
        assert score_floor_penalty([Tile.BLUE] * n) <= 0


# ── score_wall_bonus ───────────────────────────────────────────────────────────


def test_score_wall_bonus_empty_wall_is_zero():
    wall = _empty_wall()
    assert score_wall_bonus(wall) == 0


def test_score_wall_bonus_partial_row_is_zero():
    wall = _empty_wall()
    wall[0][0] = Tile.BLUE
    wall[0][1] = Tile.YELLOW
    assert score_wall_bonus(wall) == 0


def test_score_wall_bonus_complete_row_scores_two():
    wall = _empty_wall()
    for column in range(BOARD_SIZE):
        wall[0][column] = WALL_PATTERN[0][column]
    assert score_wall_bonus(wall) == BONUS_ROW


def test_score_wall_bonus_two_complete_rows_scores_four():
    wall = _empty_wall()
    for column in range(BOARD_SIZE):
        wall[0][column] = WALL_PATTERN[0][column]
        wall[1][column] = WALL_PATTERN[1][column]
    assert score_wall_bonus(wall) == BONUS_ROW * 2


def test_score_wall_bonus_partial_column_is_zero():
    wall = _empty_wall()
    wall[0][0] = WALL_PATTERN[0][0]
    wall[1][0] = WALL_PATTERN[1][0]
    assert score_wall_bonus(wall) == 0


def test_score_wall_bonus_complete_column_scores_seven():
    wall = _empty_wall()
    for row in range(BOARD_SIZE):
        wall[row][0] = WALL_PATTERN[row][0]
    assert score_wall_bonus(wall) == BONUS_COLUMN


def test_score_wall_bonus_two_complete_columns_scores_fourteen():
    wall = _empty_wall()
    for row in range(BOARD_SIZE):
        wall[row][0] = WALL_PATTERN[row][0]
        wall[row][1] = WALL_PATTERN[row][1]
    assert score_wall_bonus(wall) == BONUS_COLUMN * 2


def test_score_wall_bonus_partial_color_is_zero():
    wall = _empty_wall()
    wall[0][0] = Tile.BLUE
    assert score_wall_bonus(wall) == 0


def test_score_wall_bonus_complete_color_scores_ten():
    wall = _empty_wall()
    for row in range(BOARD_SIZE):
        column = WALL_PATTERN[row].index(Tile.BLUE)
        wall[row][column] = Tile.BLUE
    assert score_wall_bonus(wall) == BONUS_TILE


def test_score_wall_bonus_row_and_column_overlap_counts_both():
    wall = _empty_wall()
    for column in range(BOARD_SIZE):
        wall[0][column] = WALL_PATTERN[0][column]
    for row in range(BOARD_SIZE):
        wall[row][0] = WALL_PATTERN[row][0]
    assert score_wall_bonus(wall) == BONUS_ROW + BONUS_COLUMN


def test_score_wall_bonus_all_three_bonuses():
    wall = _empty_wall()
    for column in range(BOARD_SIZE):
        wall[0][column] = WALL_PATTERN[0][column]
    for row in range(BOARD_SIZE):
        wall[row][0] = WALL_PATTERN[row][0]
    for row in range(BOARD_SIZE):
        column = WALL_PATTERN[row].index(Tile.BLUE)
        wall[row][column] = Tile.BLUE
    assert score_wall_bonus(wall) == BONUS_ROW + BONUS_COLUMN + BONUS_TILE


# ── pending_placement_details ──────────────────────────────────────────────────


def test_pending_placement_details_empty_board_returns_empty():
    board = Board()
    details, _ = pending_placement_details(board)
    assert details == []


def test_pending_placement_details_returns_placement_detail_instances():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    details, _ = pending_placement_details(board)
    assert len(details) == 1
    assert isinstance(details[0], PlacementDetail)


def test_pending_placement_details_correct_row_and_column():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    details, _ = pending_placement_details(board)
    assert details[0].row == 0
    assert details[0].column == COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]


def test_pending_placement_details_lone_tile_scores_one():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    details, _ = pending_placement_details(board)
    assert details[0].placement_points == 1


def test_pending_placement_details_does_not_include_incomplete_lines():
    board = Board()
    board.pattern_lines[1] = [Tile.BLUE]  # row 1 capacity is 2 — not full
    details, _ = pending_placement_details(board)
    assert details == []


def test_pending_placement_details_returns_temp_wall():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    details, temp_wall = pending_placement_details(board)
    column = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert temp_wall[0][column] == Tile.BLUE


def test_pending_placement_details_does_not_mutate_board_wall():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    pending_placement_details(board)
    column = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert board.wall[0][column] is None


def test_pending_placement_details_second_placement_sees_first():
    """Second pending placement sees the first in the temp wall.

    YELLOW lands in column 2 of row 1; BLUE lands in column 2 of row 2.
    Both are pending — neither is on the real wall yet. Row 2's placement
    should see row 1's tile directly above it in the temp wall, scoring 2
    instead of 1.
    """
    board = Board()
    assert COLUMN_FOR_TILE_IN_ROW[Tile.YELLOW][1] == 2
    assert COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][2] == 2
    board.pattern_lines[1] = [Tile.YELLOW, Tile.YELLOW]
    board.pattern_lines[2] = [Tile.BLUE, Tile.BLUE, Tile.BLUE]
    details, _ = pending_placement_details(board)
    assert details[1].placement_points == 2


# ── pending_bonus_details ──────────────────────────────────────────────────────


def test_pending_bonus_details_empty_wall_returns_empty():
    assert pending_bonus_details(_empty_wall()) == []


def test_pending_bonus_details_returns_bonus_detail_instances():
    wall = _empty_wall()
    for column in range(BOARD_SIZE):
        wall[0][column] = WALL_PATTERN[0][column]
    bonuses = pending_bonus_details(wall)
    assert len(bonuses) >= 1
    assert isinstance(bonuses[0], BonusDetail)


def test_pending_bonus_details_completed_row():
    wall = _empty_wall()
    for column in range(BOARD_SIZE):
        wall[0][column] = WALL_PATTERN[0][column]
    bonuses = pending_bonus_details(wall)
    row_bonuses = [b for b in bonuses if b.bonus_type == "row"]
    assert len(row_bonuses) == 1
    assert row_bonuses[0].index == 0
    assert row_bonuses[0].bonus_points == BONUS_ROW


def test_pending_bonus_details_completed_column():
    wall = _empty_wall()
    for row in range(BOARD_SIZE):
        wall[row][0] = WALL_PATTERN[row][0]
    bonuses = pending_bonus_details(wall)
    column_bonuses = [b for b in bonuses if b.bonus_type == "column"]
    assert len(column_bonuses) == 1
    assert column_bonuses[0].index == 0
    assert column_bonuses[0].bonus_points == BONUS_COLUMN


def test_pending_bonus_details_completed_tile_color():
    wall = _empty_wall()
    for row in range(BOARD_SIZE):
        column = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][row]
        wall[row][column] = Tile.BLUE
    bonuses = pending_bonus_details(wall)
    tile_bonuses = [b for b in bonuses if b.bonus_type == "tile"]
    assert len(tile_bonuses) == 1
    assert tile_bonuses[0].index == Tile.BLUE.value
    assert tile_bonuses[0].bonus_points == BONUS_TILE


def test_pending_bonus_details_full_wall_yields_all_bonuses():
    """A fully completed wall should yield exactly 5 row, 5 column, 5 tile bonuses."""
    wall = _empty_wall()
    for tile in COLOR_TILES:
        for row in range(BOARD_SIZE):
            column = COLUMN_FOR_TILE_IN_ROW[tile][row]
            wall[row][column] = tile
    bonuses = pending_bonus_details(wall)
    assert len([b for b in bonuses if b.bonus_type == "row"]) == BOARD_SIZE
    assert len([b for b in bonuses if b.bonus_type == "column"]) == BOARD_SIZE
    assert len([b for b in bonuses if b.bonus_type == "tile"]) == BOARD_SIZE


# ── earned_score ───────────────────────────────────────────────────────────────


def test_earned_score_empty_board_is_zero():
    assert earned_score(Board()) == 0


def test_earned_score_partial_pattern_line_not_counted():
    board = Board()
    board.pattern_lines[1] = [Tile.YELLOW]  # row 1 capacity = 2 — not full
    assert earned_score(board) == 0


def test_earned_score_lone_tile_on_empty_wall():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    assert earned_score(board) == 1


def test_earned_score_full_pattern_line_with_wall_neighbor():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    board.wall[0][1] = Tile.YELLOW
    assert earned_score(board) == 2


def test_earned_score_two_full_lines_sums_both():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    board.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]
    assert earned_score(board) == 2


def test_earned_score_two_full_lines_sums_both_joining():
    board = Board()
    board.wall[1][2] = Tile.YELLOW
    board.pattern_lines[0] = [Tile.RED]  # → (0,2)
    board.pattern_lines[2] = [Tile.BLUE, Tile.BLUE, Tile.BLUE]  # → (2,2)
    assert earned_score(board) == 5


def test_earned_score_floor_penalty_only():
    board = Board()
    board.score = 4
    board.floor_line = [Tile.BLUE, Tile.RED]
    expected = 4 + FLOOR_PENALTIES[0] + FLOOR_PENALTIES[1]
    assert earned_score(board) == expected


def test_earned_score_floor_with_first_player_marker():
    board = Board()
    board.score = 4
    board.floor_line = [Tile.FIRST_PLAYER]
    assert earned_score(board) == 4 + FLOOR_PENALTIES[0]


def test_earned_score_placement_minus_floor():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]  # +1
    board.floor_line = [Tile.RED]
    assert earned_score(board) == 1 + FLOOR_PENALTIES[0]


def test_earned_score_cannot_be_negative():
    board = Board()
    board.floor_line = [Tile.BLUE] * 7
    assert earned_score(board) == 0
