# tests/test_scoring.py

"""Tests for scoring — pure functions in engine/scoring.py and Game scoring methods."""

from engine.constants import (
    BONUS_COLOR,
    BONUS_COLUMN,
    BONUS_ROW,
    Tile,
    BOARD_SIZE,
    FLOOR_PENALTIES,
)
from engine.board import Board
from engine.game import Game, WALL_PATTERN
from engine.scoring import (
    score_placement,
    score_floor_penalty,
    #    score_wall_bonus,
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
# earned_score — full pattern lines (wall placement preview)
# ---------------------------------------------------------------------------


def test_earned_score_lone_tile_on_empty_wall():
    # Row 0 capacity = 1 and it's full — lone placement scores 1
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]
    assert earned_score(board) == 1


def test_earned_score_partial_pattern_line_not_counted():
    # Row 1 capacity = 2; one tile is not full — should contribute 0
    board = Board()
    board.pattern_lines[1] = [Tile.YELLOW]
    assert earned_score(board) == 0


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


# ---------------------------------------------------------------------------
# earned_score — floor penalties
# ---------------------------------------------------------------------------


def test_earned_score_empty_board_is_zero():
    assert earned_score(Board()) == 0


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


# ---------------------------------------------------------------------------
# earned_score — end-of-game bonuses already on wall
# ---------------------------------------------------------------------------


def test_earned_score_no_bonus_for_partial_row():
    board = Board()
    board.wall[0][0] = Tile.BLUE
    board.wall[0][1] = Tile.YELLOW
    assert earned_score(board) == 0


def test_earned_score_complete_row_adds_two():
    board = Board()
    for col in range(BOARD_SIZE):
        board.wall[0][col] = WALL_PATTERN[0][col]
    assert earned_score(board) == BONUS_ROW


def test_earned_score_complete_column_adds_seven():
    board = Board()
    for row in range(BOARD_SIZE):
        board.wall[row][0] = WALL_PATTERN[row][0]
    assert earned_score(board) == BONUS_COLUMN


def test_earned_score_complete_color_adds_ten():
    board = Board()
    for row in range(BOARD_SIZE):
        col = WALL_PATTERN[row].index(Tile.BLUE)
        board.wall[row][col] = Tile.BLUE
    assert earned_score(board) == BONUS_COLOR


def test_earned_score_complete_row_and_column_adds_nine():
    board = Board()
    for col in range(BOARD_SIZE):
        board.wall[0][col] = WALL_PATTERN[0][col]
    for row in range(BOARD_SIZE):
        board.wall[row][0] = WALL_PATTERN[row][0]
    assert earned_score(board) == BONUS_ROW + BONUS_COLUMN


def test_earned_score_bonus_does_not_double_count_shared_cells():
    # Complete row 0, column 0, and Blue — all share (0,0). Total = 2 + 7 + 10.
    board = Board()
    for col in range(BOARD_SIZE):
        board.wall[0][col] = WALL_PATTERN[0][col]
    for row in range(BOARD_SIZE):
        board.wall[row][0] = WALL_PATTERN[row][0]
    for row in range(BOARD_SIZE):
        col = WALL_PATTERN[row].index(Tile.BLUE)
        board.wall[row][col] = Tile.BLUE
    assert earned_score(board) == BONUS_ROW + BONUS_COLUMN + BONUS_COLOR


# ---------------------------------------------------------------------------
# earned_score — combined
# ---------------------------------------------------------------------------


def test_earned_score_placement_minus_floor():
    board = Board()
    board.pattern_lines[0] = [Tile.BLUE]  # +1
    board.floor_line = [Tile.RED]  # FLOOR_PENALTIES[0]
    assert earned_score(board) == 1 + FLOOR_PENALTIES[0]


def test_earned_score_cannot_be_negative():
    board = Board()
    board.floor_line = [Tile.BLUE] * 7
    assert earned_score(board) == 0


# ---------------------------------------------------------------------------
# grand_total — pure function
# ---------------------------------------------------------------------------


def test_grand_total_empty_board():
    assert earned_score(Board()) == 0


def test_grand_total_sums_carried_and_earned():
    board = Board()
    board.score = 10
    board.pattern_lines[0] = [Tile.BLUE]  # +1 earned
    assert earned_score(board) == 11


def test_grand_total_with_floor_penalty():
    board = Board()
    board.score = 5
    board.floor_line = [Tile.RED]
    assert earned_score(board) == 5 + FLOOR_PENALTIES[0]


# ---------------------------------------------------------------------------
# Game.score_round — integration tests (tests mutation through Game)
# ---------------------------------------------------------------------------


def test_full_pattern_line_moves_tile_to_wall():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.wall[0][0] == Tile.BLUE


def test_completed_line_remaining_tiles_go_to_discard():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[1] = [Tile.YELLOW, Tile.YELLOW]
    game.score_round()
    assert game.state.discard.count(Tile.YELLOW) == 1


def test_incomplete_pattern_line_is_unchanged():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[2] = [Tile.RED]
    game.score_round()
    assert player.pattern_lines[2] == [Tile.RED]
    assert player.wall[2][2] is None


def test_tile_with_no_neighbours_scores_one_point():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 1


def test_tile_with_horizontal_neighbours_scores_run_length():
    game = Game()
    player = game.state.players[0]
    player.wall[0][1] = Tile.YELLOW
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 2


def test_tile_with_vertical_neighbours_scores_run_length():
    game = Game()
    player = game.state.players[0]
    player.wall[1][0] = Tile.WHITE
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 2


def test_tile_with_both_neighbours_scores_combined_run_lengths():
    game = Game()
    player = game.state.players[0]
    player.wall[0][1] = Tile.YELLOW
    player.wall[1][0] = Tile.WHITE
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 4


def test_floor_penalties_are_applied():
    game = Game()
    player = game.state.players[0]
    player.score = 10
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -1, -1, -2 = -4
    game.score_round()
    assert player.score == 6


def test_score_does_not_go_below_zero():
    game = Game()
    player = game.state.players[0]
    player.score = 1
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.score_round()
    assert player.score == 0


def test_floor_line_is_cleared_after_scoring():
    game = Game()
    player = game.state.players[0]
    player.floor_line = [Tile.BLUE, Tile.RED]
    game.score_round()
    assert player.floor_line == []


def test_score_floor_does_not_send_first_player_tile_to_discard():
    game = Game()
    player = game.state.players[0]
    player.floor_line = [Tile.FIRST_PLAYER, Tile.BLUE]
    game._score_floor(player)
    assert Tile.FIRST_PLAYER not in game.state.discard
    assert Tile.BLUE in game.state.discard


def test_player_with_first_player_tile_starts_next_round():
    game = Game()
    game.state.players[1].floor_line = [Tile.FIRST_PLAYER]
    game.score_round()
    assert game.state.current_player == 1


# ---------------------------------------------------------------------------
# Game.is_game_over
# ---------------------------------------------------------------------------


def test_game_is_not_over_with_empty_walls():
    assert Game().is_game_over() is False


def test_game_is_not_over_with_incomplete_row():
    game = Game()
    game.state.players[0].wall[0] = [Tile.BLUE, Tile.YELLOW, None, None, None]
    assert game.is_game_over() is False


def test_game_is_over_when_one_player_completes_a_row():
    game = Game()
    game.state.players[0].wall[0] = [
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
        Tile.BLACK,
        Tile.WHITE,
    ]
    assert game.is_game_over() is True


def test_game_is_over_when_second_player_completes_a_row():
    game = Game()
    game.state.players[1].wall[2] = [
        Tile.BLACK,
        Tile.WHITE,
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
    ]
    assert game.is_game_over() is True


# ---------------------------------------------------------------------------
# Game.score_game — end-of-game bonus scoring
# ---------------------------------------------------------------------------


def test_complete_row_scores_two_points():
    game = Game()
    game.state.players[0].wall[0] = [
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
        Tile.BLACK,
        Tile.WHITE,
    ]
    game.score_game()
    assert game.state.players[0].score == 2


def test_two_complete_rows_scores_four_points():
    game = Game()
    p = game.state.players[0]
    p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    p.wall[1] = [Tile.WHITE, Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK]
    game.score_game()
    assert p.score == 4


def test_complete_column_scores_seven_points():
    game = Game()
    p = game.state.players[0]
    for row in range(BOARD_SIZE):
        p.wall[row][0] = WALL_PATTERN[row][0]
    game.score_game()
    assert p.score == 7


def test_complete_color_scores_ten_points():
    game = Game()
    p = game.state.players[0]
    for row in range(BOARD_SIZE):
        col = game.wall_column_for(row=row, color=Tile.BLUE)
        p.wall[row][col] = Tile.BLUE
    game.score_game()
    assert p.score == 10


def test_score_game_combines_all_bonuses():
    game = Game()
    p = game.state.players[0]
    p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    for row in range(BOARD_SIZE):
        p.wall[row][0] = WALL_PATTERN[row][0]
    for row in range(BOARD_SIZE):
        col = game.wall_column_for(row=row, color=Tile.BLUE)
        p.wall[row][col] = Tile.BLUE
    game.score_game()
    assert p.score == 2 + 7 + 10


def test_score_game_applies_to_all_players():
    game = Game()
    for p in game.state.players:
        p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    game.score_game()
    for p in game.state.players:
        assert p.score == 2
