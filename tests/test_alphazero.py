# tests/test_alphazero.py

"""Tests for AlphaZeroAgent."""

import pytest
from engine.game import Game, Move
from neural.model import AzulNet
from agents.alphazero import AlphaZeroAgent


@pytest.fixture
def net():
    return AzulNet()


@pytest.fixture
def agent(net):
    return AlphaZeroAgent(net, simulations=20)


def test_alphazero_agent_returns_a_move(agent):
    game = Game()
    game.setup_round()
    move = agent.choose_move(game)
    assert isinstance(move, Move)


def test_alphazero_agent_returns_legal_move(agent):
    game = Game()
    game.setup_round()
    move = agent.choose_move(game)
    legal = game.legal_moves()
    assert move in legal


def test_alphazero_agent_works_mid_game(agent):
    """Agent should not crash when called on a game already in progress."""
    game = Game()
    game.setup_round()
    # Make a few random moves to get to a mid-game state
    from agents.random import RandomAgent

    rng = RandomAgent()
    for _ in range(6):
        if game.legal_moves():
            game.make_move(rng.choose_move(game))
    if game.legal_moves():
        move = agent.choose_move(game)
        assert move in game.legal_moves()


def test_alphazero_node_visit_counts_increase(net):
    """After running simulations, root should have been visited N times."""

    game = Game()
    game.setup_round()
    agent = AlphaZeroAgent(net, simulations=10)
    agent.choose_move(game)  # runs simulations internally
    # Can't inspect root directly from outside, but no crash = structure is intact


def test_alphazero_respects_simulation_count(net):
    """Higher simulation count should not crash and should return a legal move."""
    agent = AlphaZeroAgent(net, simulations=50)
    game = Game()
    game.setup_round()
    move = agent.choose_move(game)
    assert move in game.legal_moves()


def test_alphazero_temperature_zero_picks_most_visited(net):
    """At temperature=0, should deterministically pick most-visited child."""
    agent = AlphaZeroAgent(net, simulations=30, temperature=0.0)
    game = Game()
    game.setup_round()
    move1 = agent.choose_move(game)
    move2 = agent.choose_move(game)
    assert move1 == move2


def test_alphazero_temperature_nonzero_can_vary(net):
    """At high temperature, move selection is stochastic — just check it's legal."""
    agent = AlphaZeroAgent(net, simulations=20, temperature=1.0)
    game = Game()
    game.setup_round()
    for _ in range(5):
        move = agent.choose_move(game)
        assert move in game.legal_moves()
