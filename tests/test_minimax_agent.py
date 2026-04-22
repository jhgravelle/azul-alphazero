# tests/test_minimax_agent.py
from engine.game import Game
from engine.scoring import earned_score_unclamped
from agents.minimax import MinimaxAgent


def _make_game():
    game = Game()
    game.setup_round()
    return game


def _make_agent(max_depth: int = 3) -> MinimaxAgent:
    """Make a MinimaxAgent that always uses max_depth regardless of branching."""
    return MinimaxAgent(depths=(max_depth, max_depth, max_depth), thresholds=(0, 0))


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
    original_player = game.state.current_player
    original_center = list(game.state.center)
    agent = _make_agent(2)
    agent.choose_move(game)
    assert game.state.current_player == original_player
    assert game.state.center == original_center


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
        score_non_floor = earned_score_unclamped(
            child_non_floor.state.players[game.state.current_player]
        )
        score_floor = earned_score_unclamped(
            child_floor.state.players[game.state.current_player]
        )
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
