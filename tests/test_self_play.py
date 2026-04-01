# tests/test_self_play.py

"""Tests for the self-play harness."""

from agents.random import RandomAgent
from scripts.self_play import GameResult, run_game, run_series


def test_run_game_returns_game_result():
    """run_game should return a GameResult with expected fields."""
    p1, p2 = RandomAgent(), RandomAgent()
    result = run_game(p1, p2)
    assert isinstance(result, GameResult)


def test_run_game_result_has_valid_winner():
    """Winner should be 0, 1, or None for a tie."""
    p1, p2 = RandomAgent(), RandomAgent()
    result = run_game(p1, p2)
    assert result.winner in (0, 1, None)


def test_run_game_result_scores_are_non_negative():
    p1, p2 = RandomAgent(), RandomAgent()
    result = run_game(p1, p2)
    assert result.scores[0] >= 0
    assert result.scores[1] >= 0


def test_run_game_result_rounds_is_positive():
    p1, p2 = RandomAgent(), RandomAgent()
    result = run_game(p1, p2)
    assert result.rounds >= 1


def test_run_series_returns_correct_number_of_results():
    p1, p2 = RandomAgent(), RandomAgent()
    stats = run_series(p1, p2, n=5)
    assert stats.total_games == 5


def test_run_series_win_rates_sum_to_one():
    """Win rates plus tie rate should sum to 1.0."""
    p1, p2 = RandomAgent(), RandomAgent()
    stats = run_series(p1, p2, n=20)
    total = stats.win_rate_p1 + stats.win_rate_p2 + stats.tie_rate
    assert abs(total - 1.0) < 1e-9


def test_run_series_average_rounds_is_positive():
    p1, p2 = RandomAgent(), RandomAgent()
    stats = run_series(p1, p2, n=10)
    assert stats.avg_rounds > 0
