# tests/test_agents.py

"""Tests for the agent interface and RandomAgent."""

import pytest
import random
from engine.game import Game
from agents.base import Agent
from agents.random import RandomAgent


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
    assert len(set((m.source, m.color, m.destination) for m in moves)) > 1


def test_random_agent_respects_seeded_randomness():
    """With the same seed, RandomAgent should make the same choice."""
    game = Game()
    game.setup_round()
    agent = RandomAgent()
    random.seed(42)
    move_a = agent.choose_move(game)
    random.seed(42)
    move_b = agent.choose_move(game)
    assert (move_a.source, move_a.color, move_a.destination) == (
        move_b.source,
        move_b.color,
        move_b.destination,
    )
