# tests/test_alphazero.py
"""Tests for the AlphaZero agent."""

import pytest

from agents.alphazero import AlphaZeroAgent
from agents.base import Agent
from engine.game import Game
from neural.model import AzulNet
from neural.search_tree import AZNode, SearchTree

# ── Helpers ────────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    g = Game()
    g.setup_round()
    return g


def fresh_agent(simulations: int = 10) -> AlphaZeroAgent:
    return AlphaZeroAgent(AzulNet(), simulations=simulations, temperature=0.0)


# ── AZNode ─────────────────────────────────────────────────────────────────
# AZNode is defined in search_tree and re-exported from alphazero for
# backwards compatibility. Basic node tests live in test_search_tree.py.
# We just verify the import works and the key properties hold.


def test_aznode_importable_from_alphazero():
    node = AZNode(game=fresh_game())
    assert node.visits == 0
    assert node.total_value == 0.0
    assert node.children == []
    assert node.q_value == 0.0


def test_aznode_puct_score_positive_with_prior():
    node = AZNode(game=fresh_game(), prior=0.5)
    assert node.puct_score(parent_visits=10) > 0.0


# ── Construction ───────────────────────────────────────────────────────────


def test_alphazero_agent_is_an_agent():
    assert issubclass(AlphaZeroAgent, Agent)


def test_alphazero_constructs():
    fresh_agent()


def test_alphazero_has_internal_tree():
    agent = fresh_agent()
    assert isinstance(agent._tree, SearchTree)


# ── choose_move ────────────────────────────────────────────────────────────


def test_alphazero_returns_legal_move():
    game = fresh_game()
    move = fresh_agent(simulations=10).choose_move(game)
    assert move in game.legal_moves()


def test_alphazero_choose_move_does_not_mutate_game():
    game = fresh_game()
    factories_before = [list(f) for f in game.state.factories]
    fresh_agent(simulations=10).choose_move(game)
    assert [list(f) for f in game.state.factories] == factories_before


def test_alphazero_choose_move_with_external_tree():
    """Agent should use the provided tree instead of its own."""
    from neural.search_tree import make_policy_value_fn

    game = fresh_game()
    agent = fresh_agent(simulations=10)
    external_tree = SearchTree(
        policy_value_fn=make_policy_value_fn(agent.net),
        simulations=10,
        temperature=0.0,
    )
    external_tree.reset(game)
    move = agent.choose_move(game, tree=external_tree)
    assert move in game.legal_moves()


# ── get_policy_targets ─────────────────────────────────────────────────────


def test_alphazero_get_policy_targets_returns_legal_move():
    game = fresh_game()
    move, policy = fresh_agent(simulations=10).get_policy_targets(game)
    assert move in game.legal_moves()


def test_alphazero_get_policy_targets_sums_to_one():
    game = fresh_game()
    _, policy = fresh_agent(simulations=10).get_policy_targets(game)
    assert len(policy) > 0
    assert sum(p for _, p in policy) == pytest.approx(1.0, abs=0.01)


# ── advance / tree reuse ───────────────────────────────────────────────────


def test_advance_does_not_raise():
    game = fresh_game()
    agent = fresh_agent(simulations=10)
    move = agent.choose_move(game)
    agent.advance(move)  # should not raise


def test_after_advance_choose_move_still_legal():
    game = fresh_game()
    agent = fresh_agent(simulations=10)
    move = agent.choose_move(game)
    agent.advance(move)
    game.make_move(move)
    if not game.is_game_over():
        next_move = agent.choose_move(game)
        assert next_move in game.legal_moves()


def test_advance_with_external_tree():
    from neural.search_tree import make_policy_value_fn

    game = fresh_game()
    agent = fresh_agent(simulations=10)
    external_tree = SearchTree(
        policy_value_fn=make_policy_value_fn(agent.net),
        simulations=10,
        temperature=0.0,
    )
    external_tree.reset(game)
    move = agent.choose_move(game, tree=external_tree)
    agent.advance(move, tree=external_tree)
    assert external_tree._root is not None
    assert external_tree._root.move == move


# ── reset_tree ─────────────────────────────────────────────────────────────


def test_reset_tree_clears_root():
    game = fresh_game()
    agent = fresh_agent(simulations=10)
    agent.choose_move(game)
    agent.reset_tree(game)
    # After reset the root should reflect the new game
    assert agent._tree._root is not None


def test_alphazero_choose_move_returns_legal_move_after_advance():
    """After a move is made and the tree advanced, the next move must still be legal.

    Regression test: a stale tree that isn't advanced keeps recommending
    moves that are no longer legal.
    """
    from neural.model import AzulNet
    from agents.alphazero import AlphaZeroAgent
    from engine.game import Game

    net = AzulNet()
    agent = AlphaZeroAgent(net, simulations=5, temperature=0.0)
    game = Game()
    game.setup_round()

    for _ in range(10):
        if game.is_game_over():
            break
        legal_before = game.legal_moves()
        move = agent.choose_move(game)
        assert move in legal_before, f"Move {move} not in legal moves"
        game.make_move(move)
        game.advance()
        # Advance the agent's internal tree
        agent.advance(move)
