# tests/agents/test_alphabeta.py
from engine.game import Game
from agents.alphabeta import AlphaBetaAgent
from agents.minimax import MinimaxAgent
import pytest


def _make_game() -> Game:
    game = Game()
    game.setup_round()
    return game


def _deterministic_agent(depth: int = 3, threshold: int = 999) -> AlphaBetaAgent:
    """AlphaBetaAgent with exploration disabled and threshold high enough that
    depth is always used — makes tests deterministic and predictable."""
    return AlphaBetaAgent(depth=depth, threshold=threshold, exploration_temperature=0.0)


def test_alphabeta_returns_legal_move():
    game = _make_game()
    agent = _deterministic_agent()
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_alphabeta_does_not_mutate_game():
    game = _make_game()
    original_player_index = game.current_player_index
    original_center = list(game.center)
    agent = _deterministic_agent()
    agent.choose_move(game)
    assert game.current_player_index == original_player_index
    assert game.center == original_center


def test_alphabeta_completes_game():
    game = _make_game()
    agent = _deterministic_agent()
    moves = 0
    while not game.is_game_over():
        move = agent.choose_move(game)
        game.make_move(move)
        game.advance()
        moves += 1
        assert moves < 200, "game took too long"


def test_alphabeta_deterministic_picks_best_scored_move():
    """With exploration_temperature=0.0, AB always picks the highest-scored move."""
    game = _make_game()
    agent = _deterministic_agent(depth=2)
    move = agent.choose_move(game)
    best_move = max(agent._root_move_scores, key=lambda pair: pair[1])[0]
    assert move == best_move


def test_alphabeta_stochastic_returns_legal_move():
    """With exploration enabled, AB still returns a legal move."""
    game = _make_game()
    agent = AlphaBetaAgent(depth=2, threshold=6, exploration_temperature=0.3)
    for _ in range(10):
        move = agent.choose_move(game)
        assert move in game.legal_moves()


def test_alphabeta_policy_distribution_sums_to_one():
    game = _make_game()
    agent = _deterministic_agent(depth=2)
    agent.choose_move(game)
    dist = agent.policy_distribution(game)
    probs = [prob for _, prob in dist]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_alphabeta_policy_distribution_covers_all_legal_moves():
    game = _make_game()
    agent = _deterministic_agent(depth=2)
    agent.choose_move(game)
    dist = agent.policy_distribution(game)
    legal = game.legal_moves()
    assert len(dist) == len(legal)


def test_alphabeta_agrees_with_minimax():
    """With exploration disabled and matching depth/threshold, AB and Minimax
    choose the same move — both are deterministic greedy over scored moves."""
    game = _make_game()
    ab = AlphaBetaAgent(depth=2, threshold=999, exploration_temperature=0.0)
    mm = MinimaxAgent(depth=2, threshold=999)
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


@pytest.mark.slow
def test_alphabeta_agrees_with_minimax_deep():
    """Deeper agreement check between AB and Minimax."""
    game = _make_game()
    ab = AlphaBetaAgent(depth=3, threshold=999, exploration_temperature=0.0)
    mm = MinimaxAgent(depth=3, threshold=999)
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
