# tests/test_alphazero.py

"""Tests for the AlphaZero MCTS agent."""

import pytest

from agents.alphazero import AlphaZeroAgent, AZNode
from agents.base import Agent
from engine.game import Game
from neural.model import AzulNet


# ── Helpers ────────────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    g = Game()
    g.setup_round()
    return g


def fresh_agent(simulations: int = 10) -> AlphaZeroAgent:
    return AlphaZeroAgent(AzulNet(), simulations=simulations, temperature=0.0)


# ── AZNode structure ───────────────────────────────────────────────────────────


def test_aznode_initial_visits_zero():
    node = AZNode(game=fresh_game())
    assert node.visits == 0


def test_aznode_initial_total_value_zero():
    node = AZNode(game=fresh_game())
    assert node.total_value == 0.0


def test_aznode_children_start_empty():
    node = AZNode(game=fresh_game())
    assert node.children == []


def test_aznode_untried_moves_none_before_expand():
    node = AZNode(game=fresh_game())
    assert node._untried_moves is None


def test_aznode_untried_priors_none_before_expand():
    node = AZNode(game=fresh_game())
    assert node._untried_priors is None


def test_aznode_is_not_fully_expanded_before_expand():
    node = AZNode(game=fresh_game())
    assert not node.is_fully_expanded


def test_aznode_q_value_zero_when_unvisited():
    node = AZNode(game=fresh_game())
    assert node.q_value == 0.0


def test_aznode_q_value_correct_when_visited():
    node = AZNode(game=fresh_game())
    node.visits = 4
    node.total_value = 2.0
    assert node.q_value == pytest.approx(0.5)


def test_aznode_puct_score_unvisited_child():
    node = AZNode(game=fresh_game(), prior=0.5)
    # Unvisited child: Q=0, U = C * 0.5 * sqrt(10) / 1 > 0
    score = node.puct_score(parent_visits=10)
    assert score > 0.0


# ── _expand (lazy) ─────────────────────────────────────────────────────────────


def test_expand_stores_untried_moves():
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    assert node._untried_moves is not None
    assert len(node._untried_moves) > 0


def test_expand_stores_untried_priors():
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    assert node._untried_priors is not None
    assert node._untried_moves is not None
    assert len(node._untried_priors) == len(node._untried_moves)


def test_expand_creates_no_children():
    """Lazy expansion — children are created by _select, not _expand."""
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    assert node.children == []


def test_expand_is_idempotent():
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    assert node._untried_moves is not None
    moves_after_first = list(node._untried_moves)
    agent._expand(node)
    assert node._untried_moves == moves_after_first


def test_expand_marks_node_with_no_legal_moves_as_fully_expanded():
    """A node with no legal moves available is marked fully expanded."""
    agent = fresh_agent()
    game = Game()
    game.setup_round()
    # Clear all sources so there are no legal moves.
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    node = AZNode(game=game)
    agent._expand(node)
    assert node.is_fully_expanded


# ── _select creates children lazily ───────────────────────────────────────────


def test_select_creates_one_child_on_first_call():
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    agent._select(node)
    assert len(node.children) == 1


def test_select_creates_second_child_on_second_call():
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    agent._select(node)
    agent._select(node)
    assert len(node.children) == 2


def test_select_child_has_correct_prior():
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    child = agent._select(node)
    assert 0.0 <= child.prior <= 1.0


def test_select_child_game_is_independent_of_parent():
    agent = fresh_agent()
    node = AZNode(game=fresh_game())
    agent._expand(node)
    child = agent._select(node)
    # Mutating child game must not affect parent
    child.game.state.current_player = 99
    assert node.game.state.current_player != 99


def test_select_fully_expanded_node_picks_by_puct():
    """Once all children are created, _select picks the best PUCT child."""
    agent = fresh_agent(simulations=5)
    node = AZNode(game=fresh_game())
    agent._expand(node)
    # Exhaust all untried moves
    while node._untried_moves:
        agent._select(node)
    # Give one child many visits and high value so it dominates
    best = node.children[0]
    best.visits = 100
    best.total_value = 90.0
    node.visits = 100
    selected = agent._select(node)
    assert selected is best


# ── choose_move contract ───────────────────────────────────────────────────────


def test_alphazero_agent_is_an_agent():
    assert issubclass(AlphaZeroAgent, Agent)


def test_alphazero_returns_legal_move():
    game = fresh_game()
    agent = fresh_agent(simulations=10)
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_alphazero_choose_move_does_not_mutate_game():
    game = fresh_game()
    factories_before = [list(f) for f in game.state.factories]
    agent = fresh_agent(simulations=10)
    agent.choose_move(game)
    assert [list(f) for f in game.state.factories] == factories_before


def test_alphazero_get_policy_targets_returns_move_and_probs():
    game = fresh_game()
    agent = fresh_agent(simulations=10)
    move, policy = agent.get_policy_targets(game)
    assert move in game.legal_moves()
    assert len(policy) > 0
    total = sum(p for _, p in policy)
    assert total == pytest.approx(1.0, abs=0.01)
