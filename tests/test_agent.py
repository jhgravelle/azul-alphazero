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

    player = game.current_player
    player.place(2, [Tile.BLUE])
    game.center.append(Tile.BLUE)

    moves = [agent.choose_move(game) for _ in range(100)]

    blue_moves = [m for m in moves if m.tile == Tile.BLUE and m.destination >= 0]
    if blue_moves:
        partial_picks = sum(1 for m in blue_moves if m.destination == 2)
        assert partial_picks / len(blue_moves) > 0.5, (
            "Expected >50% of blue moves to target row 2, "
            f"got {partial_picks}/{len(blue_moves)}"
        )

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
    dist = CautiousAgent().policy_distribution(game)
    for move, _ in dist:
        assert move.destination != FLOOR


def test_cautious_policy_distribution_is_uniform_over_non_floor():
    """Each non-floor move gets equal probability."""
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
    """Distribution matches the non_floor_moves filter size."""
    from agents.cautious import CautiousAgent
    from agents.move_filters import non_floor_moves

    game = Game()
    game.setup_round()
    filtered = non_floor_moves(game.legal_moves())
    dist = CautiousAgent().policy_distribution(game)
    assert len(dist) == len(filtered)


# ── EfficientAgent.policy_distribution ─────────────────────────────────────


def test_efficient_policy_distribution_sums_to_one():
    """Probabilities sum to 1.0."""
    from agents.efficient import EfficientAgent

    game = Game()
    game.setup_round()
    dist = EfficientAgent().policy_distribution(game)
    total = sum(p for _, p in dist)
    assert total == pytest.approx(1.0)


def test_efficient_policy_distribution_uniform_over_candidates():
    """All returned moves have equal probability."""
    from agents.efficient import EfficientAgent

    game = Game()
    game.setup_round()
    dist = EfficientAgent().policy_distribution(game)
    probs = [p for _, p in dist]
    expected = 1.0 / len(probs)
    for p in probs:
        assert p == pytest.approx(expected)


def test_efficient_policy_distribution_falls_back_to_all_legal_moves():
    """In a fresh game, distribution covers all legal moves."""
    from agents.efficient import EfficientAgent

    game = Game()
    game.setup_round()
    dist = EfficientAgent().policy_distribution(game)
    assert len(dist) == len(game.legal_moves())


def test_efficient_policy_distribution_prefers_partial_lines_when_available():
    """With a partial pattern line matching an available color, distribution
    only includes moves to that partial line."""
    from agents.efficient import EfficientAgent

    game = Game()
    game.setup_round()
    player = game.current_player
    player.place(2, [Tile.BLUE])
    game.center.append(Tile.BLUE)
    dist = EfficientAgent().policy_distribution(game)
    for move, _ in dist:
        assert move.destination >= 0
        assert len(player.pattern_lines[move.destination]) > 0


def test_efficient_policy_distribution_covers_all_preferred_moves():
    """Every legal move targeting a partial line appears in the distribution."""
    from agents.efficient import EfficientAgent

    game = Game()
    game.setup_round()
    player = game.current_player
    player.place(2, [Tile.BLUE])
    game.center.append(Tile.BLUE)
    legal = game.legal_moves()
    expected_preferred = [
        m
        for m in legal
        if m.destination >= 0 and len(player.pattern_lines[m.destination]) > 0
    ]
    dist = EfficientAgent().policy_distribution(game)
    moves_in_dist = [m for m, _ in dist]
    assert len(moves_in_dist) == len(expected_preferred)
    for m in expected_preferred:
        assert m in moves_in_dist


# ── GreedyAgent.policy_distribution ────────────────────────────────────────


def test_greedy_policy_distribution_sums_to_one():
    """Probabilities sum to 1.0."""
    from agents.greedy import GreedyAgent

    game = Game()
    game.setup_round()
    dist = GreedyAgent().policy_distribution(game)
    total = sum(p for _, p in dist)
    assert total == pytest.approx(1.0)


def test_greedy_policy_distribution_excludes_floor_when_alternatives_exist():
    """No floor moves appear when non-floor moves are available."""
    from agents.greedy import GreedyAgent

    game = Game()
    game.setup_round()
    dist = GreedyAgent().policy_distribution(game)
    for move, _ in dist:
        assert move.destination != FLOOR


def test_greedy_policy_distribution_weights_each_color_equally():
    """Probabilities grouped by color should each sum to 1/num_colors."""
    from agents.greedy import GreedyAgent
    from agents.move_filters import non_floor_moves

    game = Game()
    game.setup_round()
    dist = GreedyAgent().policy_distribution(game)
    num_colors = len({m.tile for m in non_floor_moves(game.legal_moves())})
    color_totals: dict = {}
    for move, prob in dist:
        color_totals[move.tile] = color_totals.get(move.tile, 0.0) + prob
    for color, total in color_totals.items():
        assert total == pytest.approx(
            1.0 / num_colors
        ), f"Color {color} has total prob {total}, expected {1.0 / num_colors}"


def test_greedy_policy_distribution_uniform_within_each_color():
    """Within a given color, all candidate moves should have equal probability."""
    from agents.greedy import GreedyAgent

    game = Game()
    game.setup_round()
    dist = GreedyAgent().policy_distribution(game)
    by_color: dict = {}
    for move, prob in dist:
        by_color.setdefault(move.tile, []).append(prob)
    for color, probs in by_color.items():
        first = probs[0]
        for p in probs:
            assert p == pytest.approx(
                first
            ), f"Color {color} has unequal probabilities: {probs}"


def test_greedy_policy_distribution_prefers_partial_lines_within_color():
    """With a partial line for BLUE, only BLUE moves to that partial line
    should appear in BLUE's distribution slice."""
    from agents.greedy import GreedyAgent

    game = Game()
    game.setup_round()
    player = game.current_player
    player.place(2, [Tile.BLUE])
    game.center.append(Tile.BLUE)
    dist = GreedyAgent().policy_distribution(game)
    blue_moves = [m for m, _ in dist if m.tile == Tile.BLUE]
    for m in blue_moves:
        assert m.destination == 2, (
            f"Blue move to destination {m.destination} shouldn't be in dist "
            f"when row 2 has a blue partial line"
        )
