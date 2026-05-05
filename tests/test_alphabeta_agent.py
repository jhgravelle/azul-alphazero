# tests/test_alphabeta_agent.py
from engine.game import Game
from agents.alphabeta import AlphaBetaAgent
from agents.minimax import MinimaxAgent
import pytest


def _make_game():
    game = Game()
    game.setup_round()
    return game


def _make_agent(max_depth: int = 3) -> AlphaBetaAgent:
    """Make an AlphaBetaAgent that always uses max_depth regardless of branching."""
    return AlphaBetaAgent(depths=(max_depth, max_depth, max_depth), thresholds=(0, 0))


def test_alphabeta_returns_legal_move():
    game = _make_game()
    agent = _make_agent(3)
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_alphabeta_does_not_mutate_game():
    game = _make_game()
    original_player = game.current_player
    original_center = list(game.center)
    agent = _make_agent(3)
    agent.choose_move(game)
    assert game.current_player == original_player
    assert game.center == original_center


def test_alphabeta_completes_game():
    game = _make_game()
    agent = _make_agent(3)
    moves = 0
    while not game.is_game_over():
        move = agent.choose_move(game)
        game.make_move(move)
        game.advance()
        moves += 1
        assert moves < 200, "game took too long"


def test_alphabeta_agrees_with_minimax_depth2():
    """At depth 2, alpha-beta and minimax should choose the same move."""
    game = _make_game()
    ab = AlphaBetaAgent(depths=(2, 2, 2), thresholds=(0, 0))
    mm = MinimaxAgent(depths=(2, 2, 2), thresholds=(0, 0))
    for _ in range(5):
        if game.is_game_over():
            break
        move_ab = ab.choose_move(game)
        move_mm = mm.choose_move(game)
        assert (
            move_ab == move_mm
        ), f"AlphaBeta chose {move_ab} but Minimax chose {move_mm}"
        game.make_move(move_ab)
        game.advance()


def test_alphabeta_policy_distribution_is_uniform():
    game = _make_game()
    agent = _make_agent(2)
    dist = agent.policy_distribution(game)
    legal = game.legal_moves()
    assert len(dist) == len(legal)
    probs = [p for _, p in dist]
    assert abs(sum(probs) - 1.0) < 1e-6


@pytest.mark.slow
def test_alphabeta_agrees_with_minimax_depth3():
    """At depth 3, alpha-beta and minimax should choose the same move."""
    game = _make_game()
    ab = AlphaBetaAgent(depths=(3, 3, 3), thresholds=(0, 0))
    mm = MinimaxAgent(depths=(3, 3, 3), thresholds=(0, 0))
    for _ in range(10):
        if game.is_game_over():
            break
        move_ab = ab.choose_move(game)
        move_mm = mm.choose_move(game)
        assert (
            move_ab == move_mm
        ), f"AlphaBeta chose {move_ab} but Minimax chose {move_mm}"
        game.make_move(move_ab)
        game.advance()
