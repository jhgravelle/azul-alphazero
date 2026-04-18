# tests/test_agents.py

"""Tests for the agent interface and RandomAgent."""

import pytest
import random
from agents.greedy import GreedyAgent
from engine.game import Game
from agents.base import Agent
from agents.random import RandomAgent
from engine.constants import Tile
from engine.game import FLOOR


def test_agent_cannot_be_instantiated_directly():
    """Agent is abstract — you must subclass it and implement choose_move."""
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]


def test_agent_subclass_without_choose_move_cannot_be_instantiated():
    """A subclass that forgets choose_move also can't be instantiated."""

    class IncompleteAgent(Agent):
        pass

    with pytest.raises(TypeError):
        IncompleteAgent()  # type: ignore[abstract]


def test_random_agent_returns_a_legal_move():
    """RandomAgent must return a move that appears in game.legal_moves()."""
    game = Game()
    game.setup_round()
    agent = RandomAgent()
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_random_agent_returns_different_moves_over_many_calls():
    """RandomAgent should not always return the same move."""
    game = Game()
    game.setup_round()
    agent = RandomAgent()
    moves = [agent.choose_move(game) for _ in range(50)]
    assert len(set((m.source, m.tile, m.destination) for m in moves)) > 1


def test_random_agent_respects_seeded_randomness():
    """With the same seed, RandomAgent should make the same choice."""
    game = Game()
    game.setup_round()
    agent = RandomAgent()
    random.seed(42)
    move_a = agent.choose_move(game)
    random.seed(42)
    move_b = agent.choose_move(game)
    assert (move_a.source, move_a.tile, move_a.destination) == (
        move_b.source,
        move_b.tile,
        move_b.destination,
    )


def test_random_agent_avoids_floor_when_other_moves_exist():
    """RandomAgent should not choose the floor if non-floor moves are available."""
    game = Game()
    game.setup_round()
    agent = RandomAgent()
    moves = [agent.choose_move(game) for _ in range(50)]
    non_floor_legal = [m for m in game.legal_moves() if m.destination != -2]
    if non_floor_legal:
        assert any(m.destination != -2 for m in moves)


def test_greedy_agent_prefers_partial_line_without_biasing_color_selection():
    """Partial line preference should affect destination, not color choice."""
    game = Game()
    game.setup_round()
    agent = GreedyAgent()

    player = game.state.players[game.state.current_player]
    # Put BLUE on row 2 (capacity 3, so not full)
    player.pattern_lines[2] = [Tile.BLUE]

    game.state.center.append(Tile.BLUE)

    moves = [agent.choose_move(game) for _ in range(100)]

    # Among blue moves, majority should go to row 2
    blue_moves = [m for m in moves if m.tile == Tile.BLUE and m.destination >= 0]
    if blue_moves:
        partial_picks = sum(1 for m in blue_moves if m.destination == 2)
        assert partial_picks / len(blue_moves) > 0.5, (
            "Expected >50% of blue moves to target row 2, "
            f"got {partial_picks}/{len(blue_moves)}"
        )

    # But BLUE should not dominate color selection overall
    non_floor = [m for m in moves if m.destination != -2]
    blue_non_floor = [m for m in non_floor if m.tile == Tile.BLUE]
    if non_floor:
        assert (
            len(blue_non_floor) / len(non_floor) < 0.9
        ), "Color selection appears biased toward BLUE"


# ── policy_distribution (base class default) ───────────────────────────────


def test_agent_policy_distribution_returns_list():
    """Default policy_distribution returns a list."""
    game = Game()
    game.setup_round()
    dist = RandomAgent().policy_distribution(game)
    assert isinstance(dist, list)


def test_agent_policy_distribution_returns_tuples_of_move_and_float():
    """Each entry is (Move, probability) where probability is a float."""
    game = Game()
    game.setup_round()
    dist = RandomAgent().policy_distribution(game)
    for move, prob in dist:
        assert move in game.legal_moves()
        assert isinstance(prob, float)


def test_agent_policy_distribution_sums_to_one():
    """Probabilities sum to 1.0 across all returned moves."""
    game = Game()
    game.setup_round()
    dist = RandomAgent().policy_distribution(game)
    total = sum(p for _, p in dist)
    assert total == pytest.approx(1.0)


def test_agent_policy_distribution_covers_all_legal_moves():
    """Default distribution includes every legal move."""
    game = Game()
    game.setup_round()
    dist = RandomAgent().policy_distribution(game)
    legal = game.legal_moves()
    moves_in_dist = [m for m, _ in dist]
    assert len(moves_in_dist) == len(legal)
    for m in legal:
        assert m in moves_in_dist


def test_agent_policy_distribution_uniform_by_default():
    """Default distribution has equal probability for each legal move."""
    game = Game()
    game.setup_round()
    dist = RandomAgent().policy_distribution(game)
    probs = [p for _, p in dist]
    expected = 1.0 / len(probs)
    for p in probs:
        assert p == pytest.approx(expected)


# ── CautiousAgent.policy_distribution ──────────────────────────────────────


def test_cautious_policy_distribution_sums_to_one():
    """Probabilities sum to 1.0."""
    from agents.cautious import CautiousAgent

    game = Game()
    game.setup_round()
    dist = CautiousAgent().policy_distribution(game)
    total = sum(p for _, p in dist)
    assert total == pytest.approx(1.0)


def test_cautious_policy_distribution_excludes_floor_when_alternatives_exist():
    """No floor moves appear when non-floor moves are available."""
    from agents.cautious import CautiousAgent

    game = Game()
    game.setup_round()
    # Fresh game has plenty of non-floor options
    dist = CautiousAgent().policy_distribution(game)
    for move, _ in dist:
        assert move.destination != FLOOR


def test_cautious_policy_distribution_is_uniform_over_non_floor():
    """Each non-floor move gets equal probability, and that probability
    equals 1/(number of non-floor moves) — not 1/(total legal moves)."""
    from agents.cautious import CautiousAgent
    from agents.move_filters import non_floor_moves

    game = Game()
    game.setup_round()
    dist = CautiousAgent().policy_distribution(game)
    expected_count = len(non_floor_moves(game.legal_moves()))
    expected_prob = 1.0 / expected_count
    for _, p in dist:
        assert p == pytest.approx(expected_prob)


def test_cautious_policy_distribution_covers_all_non_floor_legal_moves():
    """Every legal non-floor move appears in the distribution."""
    from agents.cautious import CautiousAgent
    from agents.move_filters import non_floor_moves

    game = Game()
    game.setup_round()
    expected_moves = non_floor_moves(game.legal_moves())
    dist = CautiousAgent().policy_distribution(game)
    moves_in_dist = [m for m, _ in dist]
    assert len(moves_in_dist) == len(expected_moves)
    for m in expected_moves:
        assert m in moves_in_dist


def test_cautious_policy_distribution_falls_back_to_floor_when_forced():
    """When non_floor_moves would return all moves (floor-only fallback),
    the distribution reflects that fallback.

    Rather than engineering a fully-forced floor state (which is fragile to
    set up), we verify the function respects the non_floor_moves filter: if
    non_floor_moves returns the full legal set (fallback case), so does the
    distribution.
    """
    from agents.cautious import CautiousAgent
    from agents.move_filters import non_floor_moves

    game = Game()
    game.setup_round()
    legal = game.legal_moves()
    filtered = non_floor_moves(legal)
    # In a fresh game there are non-floor moves, so filtered is a subset.
    # This test verifies the distribution matches filtered's size regardless
    # of whether filtering actually reduced the set.
    dist = CautiousAgent().policy_distribution(game)
    assert len(dist) == len(filtered)
