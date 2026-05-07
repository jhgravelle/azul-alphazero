# tests/test_minimax_agent.py
from engine.game import Game
from agents.minimax import MinimaxAgent


def _make_game():
    game = Game()
    game.setup_round()
    return game


def _make_agent(max_depth: int = 2) -> MinimaxAgent:
    """Make a MinimaxAgent that always uses max_depth regardless of branching."""
    return MinimaxAgent(depth=max_depth, threshold=999)


def test_minimax_agent_returns_legal_move():
    game = _make_game()
    agent = _make_agent(2)
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_minimax_agent_depth_1_picks_legal_move():
    game = _make_game()
    agent = _make_agent(1)
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_minimax_agent_does_not_mutate_game():
    game = _make_game()
    original_player_index = game.current_player_index
    original_center = list(game.center)
    agent = _make_agent(2)
    agent.choose_move(game)
    assert game.current_player_index == original_player_index
    assert game.center == original_center


def test_minimax_agent_completes_game():
    """Agent can play a full game without crashing."""
    game = _make_game()
    agent = _make_agent(2)
    moves = 0
    while not game.is_game_over():
        move = agent.choose_move(game)
        game.make_move(move)
        game.advance()
        moves += 1
        assert moves < 200, "game took too long"


def test_minimax_agent_avoids_floor():
    """Agent prefers moves that don't add floor tiles when a pattern
    line move is available with equal or better immediate score."""
    game = _make_game()
    agent = _make_agent(1)
    move = agent.choose_move(game)
    legal = game.legal_moves()
    non_floor = [m for m in legal if m.destination != -2]
    floor_moves = [m for m in legal if m.destination == -2]
    if non_floor and floor_moves:
        child_non_floor = game.clone()
        child_non_floor.make_move(non_floor[0])
        child_floor = game.clone()
        child_floor.make_move(floor_moves[0])
        score_non_floor = child_non_floor.players[game.current_player_index].earned
        score_floor = child_floor.players[game.current_player_index].earned
        if score_non_floor >= score_floor:
            assert move.destination != -2


def test_minimax_agent_policy_distribution_is_uniform():
    game = _make_game()
    agent = _make_agent(2)
    dist = agent.policy_distribution(game)
    legal = game.legal_moves()
    assert len(dist) == len(legal)
    probs = [p for _, p in dist]
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(abs(p - probs[0]) < 1e-6 for p in probs)
