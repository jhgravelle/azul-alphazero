# tests/test_board.py

from engine.constants import BOARD_SIZE
from engine.board import Board


def test_player_board_initial_score_is_zero():
    board = Board()
    assert board.score == 0


def test_player_board_pattern_lines_are_empty():
    board = Board()
    assert len(board.pattern_lines) == BOARD_SIZE
    for row in board.pattern_lines:
        assert row == []


def test_player_board_wall_is_empty():
    board = Board()
    assert len(board.wall) == BOARD_SIZE
    for row in board.wall:
        assert len(row) == BOARD_SIZE
        for cell in row:
            assert cell is None


def test_player_board_floor_line_is_empty():
    """The floor line starts empty."""
    board = Board()
    assert board.floor_line == []


def test_player_board_initial_clamped_points_is_zero():
    board = Board()
    assert board.clamped_points == 0


def test_clamped_points_stays_zero_when_score_does_not_go_negative():
    board = Board()
    board.score = 10
    board.clamped_points = 0
    # No clamping needed — score stays positive
    assert board.clamped_points == 0


def test_clamped_points_and_score_are_independent_fields():
    board = Board()
    board.score = 0
    board.clamped_points = 6
    assert board.score == 0
    assert board.clamped_points == 6
