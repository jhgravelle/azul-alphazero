# tests/agents/test_minimax.py
from engine.game import Game
from agents.minimax import MinimaxAgent


def _make_game() -> Game:
    game = Game()
    game.setup_round()
    return game


def _make_agent(depth: int = 2) -> MinimaxAgent:
    """MinimaxAgent with threshold high enough that depth is always used."""
    return MinimaxAgent(depth=depth, threshold=0)


def test_minimax_returns_legal_move():
    game = _make_game()
    agent = _make_agent(2)
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_minimax_depth_1_returns_legal_move():
    game = _make_game()
    agent = _make_agent(1)
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_minimax_does_not_mutate_game():
    game = _make_game()
    original_player_index = game.current_player_index
    original_center = list(game.center)
    agent = _make_agent(2)
    agent.choose_move(game)
    assert game.current_player_index == original_player_index
    assert game.center == original_center


def test_minimax_completes_game():
    game = _make_game()
    agent = _make_agent(2)
    moves = 0
    while not game.is_game_over():
        move = agent.choose_move(game)
        game.make_move(move)
        game.advance()
        moves += 1
        assert moves < 200, "game took too long"


def test_minimax_policy_distribution_is_uniform():
    game = _make_game()
    agent = _make_agent(2)
    dist = agent.policy_distribution(game)
    legal = game.legal_moves()
    assert len(dist) == len(legal)
    probs = [prob for _, prob in dist]
    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(abs(prob - probs[0]) < 1e-6 for prob in probs)


def test_minimax_avoids_floor_when_pattern_line_scores_better():
    """At depth 1, minimax picks a pattern line move over floor when it scores more."""
    game = _make_game()
    agent = _make_agent(1)
    current_player_index = game.current_player_index

    legal = game.legal_moves()
    non_floor = [m for m in legal if m.destination != -2]
    floor_moves = [m for m in legal if m.destination == -2]

    if not non_floor or not floor_moves:
        return  # can't test this position, skip

    best_non_floor_score = max(
        game.clone().make_move(m) or game.players[current_player_index].earned
        for m in non_floor
    )
    # Re-score properly via clone
    best_non_floor_score = max(
        _score_move(game, m, current_player_index) for m in non_floor
    )
    best_floor_score = max(
        _score_move(game, m, current_player_index) for m in floor_moves
    )

    if best_non_floor_score > best_floor_score:
        move = agent.choose_move(game)
        assert (
            move.destination != -2
        ), f"Expected pattern line move but got floor move {move}"


def _score_move(game: Game, move, player_index: int) -> float:
    child = game.clone()
    before = child.players[player_index].earned
    child.make_move(move)
    return child.players[player_index].earned - before
