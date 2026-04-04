# tests/test_scoring.py

"""Tests for scoring — pure functions in engine/scoring.py and Game scoring methods."""

from engine.constants import (
    Tile,
    BONUS_ROW,
    BONUS_COLUMN,
    BONUS_COLOR,
    BOARD_SIZE,
    FLOOR_PENALTIES,
    WALL_PATTERN,
)
from engine.board import Board
from engine.scoring import (
    score_placement,
    score_floor_penalty,
    score_wall_bonus,
    earned_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_wall() -> list[list[Tile | None]]:
    return [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]


# ---------------------------------------------------------------------------
# score_placement — pure function
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# score_floor_penalty — pure function
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# score_wall_bonus — pure function
# ---------------------------------------------------------------------------


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
    for col in range(BOARD_SIZE):
        wall[0][col] = WALL_PATTERN[0][col]
    assert score_wall_bonus(wall) == BONUS_ROW


def test_score_wall_bonus_two_complete_rows_scores_four():
    wall = _empty_wall()
    for col in range(BOARD_SIZE):
        wall[0][col] = WALL_PATTERN[0][col]
        wall[1][col] = WALL_PATTERN[1][col]
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
        col = WALL_PATTERN[row].index(Tile.BLUE)
        wall[row][col] = Tile.BLUE
    assert score_wall_bonus(wall) == BONUS_COLOR


def test_score_wall_bonus_row_and_column_overlap_counts_both():
    wall = _empty_wall()
    for col in range(BOARD_SIZE):
        wall[0][col] = WALL_PATTERN[0][col]
    for row in range(BOARD_SIZE):
        wall[row][0] = WALL_PATTERN[row][0]
    assert score_wall_bonus(wall) == BONUS_ROW + BONUS_COLUMN


def test_score_wall_bonus_all_three_bonuses():
    wall = _empty_wall()
    for col in range(BOARD_SIZE):
        wall[0][col] = WALL_PATTERN[0][col]
    for row in range(BOARD_SIZE):
        wall[row][0] = WALL_PATTERN[row][0]
    for row in range(BOARD_SIZE):
        col = WALL_PATTERN[row].index(Tile.BLUE)
        wall[row][col] = Tile.BLUE
    assert score_wall_bonus(wall) == BONUS_ROW + BONUS_COLUMN + BONUS_COLOR


# ---------------------------------------------------------------------------
# earned_score — full pattern lines (wall placement preview)
# ---------------------------------------------------------------------------


def test_earned_score_empty_board_is_zero():
    assert earned_score(Board()) == 0


def test_earned_score_partial_pattern_line_not_counted():
    # Row 1 capacity = 2; one tile is not full — should contribute 0
    board = Board()
    board.pattern_lines[1] = [Tile.YELLOW]
    assert earned_score(board) == 0


def test_earned_score_lone_tile_on_empty_wall():
    # Row 0 capacity = 1 and it's full — lone placement scores 1
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    assert earned_score(board) == 1


def test_earned_score_full_pattern_line_with_wall_neighbor():
    # Row 0: Blue → col 0. Yellow already at (0,1). Horizontal run = 2.
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    board.wall[0][1] = Tile.YELLOW
    assert earned_score(board) == 2


def test_earned_score_two_full_lines_sums_both():
    # Two lone placements on an otherwise empty wall → 1 + 1 = 2
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]  # → (0,0)
    board.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]  # → (2,2)
    assert earned_score(board) == 2


def test_earned_score_two_full_lines_sums_both_joining():
    # Two lone placements joined with existing wall tile → 2 + 3 = 5
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
    board.floor_line = [Tile.RED]  # FLOOR_PENALTIES[0]
    assert earned_score(board) == 1 + FLOOR_PENALTIES[0]


def test_earned_score_cannot_be_negative():
    board = Board()
    board.floor_line = [Tile.BLUE] * 7
    assert earned_score(board) == 0
