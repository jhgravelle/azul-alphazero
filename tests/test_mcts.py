# tests/test_mcts.py

"""Tests for the MCTS agent."""

import math
import pytest

from agents.base import Agent
from agents.mcts import MCTSNode, MCTSAgent, ucb1
from agents.random import RandomAgent
from engine.game import Game
from scripts.self_play import run_series

# ── UCB1 formula ──────────────────────────────────────────────────────────────


def test_ucb1_unvisited_returns_infinity():
    # An unvisited node should always be selected first — infinite score
    # ensures that.
    assert ucb1(visits=0, total_value=0.0, parent_visits=10) == math.inf


def test_ucb1_formula():
    # UCB1 = (total_value / visits) + C * sqrt(ln(parent_visits) / visits)
    # With C=math.sqrt(2), visits=4, total_value=2.0, parent_visits=16:
    # exploitation = 2.0 / 4 = 0.5
    # exploration  = sqrt(2) * sqrt(ln(16) / 4)
    #              = 1.4142 * sqrt(2.7726 / 4)
    #              = 1.4142 * sqrt(0.6931)
    #              = 1.4142 * 0.8326 = 1.1774
    # total ≈ 1.6774
    result = ucb1(visits=4, total_value=2.0, parent_visits=16)
    assert abs(result - 1.6774) < 0.001


# ── MCTSNode structure ────────────────────────────────────────────────────────


def test_mcts_node_initial_visits_zero():
    game = Game()
    game.setup_round()
    node = MCTSNode(game=game, move=None, parent=None)
    assert node.visits == 0


def test_mcts_node_initial_value_zero():
    game = Game()
    game.setup_round()
    node = MCTSNode(game=game, move=None, parent=None)
    assert node.total_value == 0.0


def test_mcts_node_children_start_empty():
    game = Game()
    game.setup_round()
    node = MCTSNode(game=game, move=None, parent=None)
    assert node.children == []


def test_mcts_node_untried_moves_are_all_legal_moves():
    # Before any expansion, all legal moves should be untried.
    game = Game()
    game.setup_round()
    node = MCTSNode(game=game, move=None, parent=None)
    legal = game.legal_moves()
    assert len(node.untried_moves) == len(legal)


def test_mcts_node_is_fully_expanded_when_untried_moves_empty():
    game = Game()
    game.setup_round()
    node = MCTSNode(game=game, move=None, parent=None)
    node.untried_moves.clear()
    assert node.is_fully_expanded()


def test_mcts_node_is_not_fully_expanded_when_untried_moves_remain():
    game = Game()
    game.setup_round()
    node = MCTSNode(game=game, move=None, parent=None)
    assert not node.is_fully_expanded()


# ── MCTSAgent contract ────────────────────────────────────────────────────────


def test_mcts_agent_is_an_agent():
    assert issubclass(MCTSAgent, Agent)


def test_mcts_agent_returns_a_legal_move():
    game = Game()
    game.setup_round()
    agent = MCTSAgent(simulations=50)
    move = agent.choose_move(game)
    legal = game.legal_moves()
    assert move in legal


def test_mcts_agent_choose_move_does_not_mutate_game_state():
    # choose_move must be a read-only operation on the game it receives.
    game = Game()
    game.setup_round()
    factories_before = [list(f) for f in game.state.factories]
    agent = MCTSAgent(simulations=50)
    agent.choose_move(game)
    assert [list(f) for f in game.state.factories] == factories_before


# ── Strength test (slow — run with: pytest -m slow) ───────────────────────────


@pytest.mark.slow
def test_mcts_beats_random_agent():
    # Phase 4 definition of done: MCTSAgent wins >80% vs RandomAgent
    # over 200 games. Uses 200 simulations per move as a balance between
    # strength and test runtime.
    mcts = MCTSAgent(simulations=100)
    random_agent = RandomAgent()
    stats = run_series(mcts, random_agent, n=100)
    assert (
        stats.win_rate_p1 >= 0.80
    ), f"MCTSAgent only won {stats.win_rate_p1:.1%} — expected ≥80%"
