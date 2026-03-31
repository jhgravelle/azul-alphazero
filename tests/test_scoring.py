# tests/test_scoring.py

"""Tests for end-of-round scoring."""

from engine.constants import BOARD_SIZE
from engine.game import Game, WALL_PATTERN
from engine.tile import Tile


def test_full_pattern_line_moves_tile_to_wall():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[0] = [Tile.BLUE]  # row 0 capacity is 1 — already full
    game.score_round()
    assert player.wall[0][0] == Tile.BLUE


def test_completed_line_remaining_tiles_go_to_discard():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[1] = [Tile.YELLOW, Tile.YELLOW]  # row 1 capacity is 2
    game.score_round()
    assert (
        game.state.discard.count(Tile.YELLOW) == 1
    )  # one goes to wall, one to discard


def test_incomplete_pattern_line_is_unchanged():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[2] = [Tile.RED]  # row 2 capacity is 3 — not full
    game.score_round()
    assert player.pattern_lines[2] == [Tile.RED]
    assert player.wall[2][2] is None  # RED belongs in col 2 of row 2


def test_tile_with_no_neighbours_scores_one_point():
    game = Game()
    player = game.state.players[0]
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 1


def test_tile_with_horizontal_neighbours_scores_run_length():
    game = Game()
    player = game.state.players[0]
    # Pre-place a tile in row 0 col 1 (YELLOW), then complete col 0 (BLUE)
    player.wall[0][1] = Tile.YELLOW
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 2  # horizontal run of 2


def test_tile_with_vertical_neighbours_scores_run_length():
    game = Game()
    player = game.state.players[0]
    # Pre-place a tile in row 1 col 0 (WHITE), then complete row 0 col 0 (BLUE)
    player.wall[1][0] = Tile.WHITE
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 2  # vertical run of 2


def test_tile_with_both_neighbours_scores_combined_run_lengths():
    game = Game()
    player = game.state.players[0]
    player.wall[0][1] = Tile.YELLOW  # horizontal neighbour
    player.wall[1][0] = Tile.WHITE  # vertical neighbour
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 4  # h-run of 2 + v-run of 2


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
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # would be -3 net
    game.score_round()
    assert player.score == 0


def test_floor_line_is_cleared_after_scoring():
    game = Game()
    player = game.state.players[0]
    player.floor_line = [Tile.BLUE, Tile.RED]
    game.score_round()
    assert player.floor_line == []


# --- is_game_over tests ---


def test_game_is_not_over_with_empty_walls():
    game = Game()
    assert game.is_game_over() is False


def test_game_is_not_over_with_incomplete_row():
    game = Game()
    player = game.state.players[0]
    player.wall[0] = [Tile.BLUE, Tile.YELLOW, None, None, None]
    assert game.is_game_over() is False


def test_game_is_over_when_one_player_completes_a_row():
    game = Game()
    player = game.state.players[0]
    player.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    assert game.is_game_over() is True


def test_game_is_over_when_second_player_completes_a_row():
    game = Game()
    player = game.state.players[1]
    player.wall[2] = [Tile.BLACK, Tile.WHITE, Tile.BLUE, Tile.YELLOW, Tile.RED]
    assert game.is_game_over() is True


# --- score_game (bonus scoring) tests ---


def test_complete_row_scores_two_points():
    game = Game()
    player = game.state.players[0]
    player.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    game.score_game()
    assert player.score == 2


def test_two_complete_rows_scores_four_points():
    game = Game()
    player = game.state.players[0]
    player.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    player.wall[1] = [Tile.WHITE, Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK]
    game.score_game()
    assert player.score == 4


def test_complete_column_scores_seven_points():
    game = Game()
    player = game.state.players[0]
    for row in range(BOARD_SIZE):
        player.wall[row][0] = WALL_PATTERN[row][0]
    game.score_game()
    assert player.score == 7


def test_complete_color_scores_ten_points():
    game = Game()
    player = game.state.players[0]
    for row in range(BOARD_SIZE):
        col = game.wall_column_for(row=row, color=Tile.BLUE)
        player.wall[row][col] = Tile.BLUE
    game.score_game()
    assert player.score == 10


def test_score_game_combines_all_bonuses():
    game = Game()
    player = game.state.players[0]
    # Complete row 0
    player.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    # Complete column 0
    for row in range(BOARD_SIZE):
        player.wall[row][0] = WALL_PATTERN[row][0]
    # Complete all blues (col 0 of row 0 is already set above)
    for row in range(BOARD_SIZE):
        col = game.wall_column_for(row=row, color=Tile.BLUE)
        player.wall[row][col] = Tile.BLUE
    game.score_game()
    assert player.score == 2 + 7 + 10  # row + column + color


def test_score_game_applies_to_all_players():
    game = Game()
    for player in game.state.players:
        player.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    game.score_game()
    for player in game.state.players:
        assert player.score == 2


def test_player_with_first_player_tile_starts_next_round():
    game = Game()
    # Player 1 (index 1) has the first player marker on their floor
    game.state.players[1].floor_line = [Tile.FIRST_PLAYER]
    game.score_round()
    assert game.state.current_player == 1
