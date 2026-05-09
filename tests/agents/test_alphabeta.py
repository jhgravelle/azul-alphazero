# tests/agents/test_alphabeta.py
from engine.game import Game
from agents.alphabeta import AlphaBetaAgent


def _make_game() -> Game:
    game = Game()
    game.setup_round()
    return game


def test_alphabeta_returns_legal_move():
    game = _make_game()
    agent = AlphaBetaAgent(depth=1, threshold=3)
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_alphabeta_does_not_mutate_game():
    game = _make_game()
    original_player_index = game.current_player_index
    original_center = list(game.center)
    agent = AlphaBetaAgent(depth=1, threshold=3)
    agent.choose_move(game)
    assert game.current_player_index == original_player_index
    assert game.center == original_center


def test_alphabeta_completes_game():
    game = _make_game()
    agent = AlphaBetaAgent(depth=1, threshold=3)
    moves = 0
    while not game.is_game_over():
        move = agent.choose_move(game)
        game.make_move(move)
        game.advance()
        moves += 1
        assert moves < 200, "game took too long"


def test_alphabeta_stochastic_returns_legal_move():
    """With exploration enabled, AB still returns a legal move."""
    game = _make_game()
    agent = AlphaBetaAgent(depth=2, threshold=6, exploration_temperature=0.3)
    for _ in range(10):
        move = agent.choose_move(game)
        assert move in game.legal_moves()


def test_alphabeta_policy_distribution_sums_to_one():
    game = _make_game()
    agent = AlphaBetaAgent(depth=1, threshold=3)
    agent.choose_move(game)
    dist = agent.policy_distribution(game)
    probs = [prob for _, prob in dist]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_alphabeta_policy_distribution_covers_all_legal_moves():
    game = _make_game()
    agent = AlphaBetaAgent(depth=1, threshold=3)
    agent.choose_move(game)
    dist = agent.policy_distribution(game)
    legal = game.legal_moves()
    assert len(dist) == len(legal)
