# tests/test_agents.py

"""Tests for the agent interface and RandomAgent."""

import pytest
import random
from engine.game import Game
from agents.base import Agent
from agents.random import RandomAgent
from engine.tile import Tile


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


def test_random_agent_avoids_floor_when_other_moves_exist():
    """RandomAgent should not choose the floor if non-floor moves are available."""
    game = Game()
    game.setup_round()
    agent = RandomAgent()
    moves = [agent.choose_move(game) for _ in range(50)]
    non_floor_legal = [m for m in game.legal_moves() if m.destination != -2]
    if non_floor_legal:
        assert any(m.destination != -2 for m in moves)


def test_random_agent_prefers_partial_line_without_biasing_color_selection():
    """Partial line preference should affect destination, not color choice."""
    game = Game()
    game.setup_round()
    agent = RandomAgent()

    player = game.state.players[game.state.current_player]
    # Put BLUE on row 2 (capacity 3, so not full)
    player.pattern_lines[2] = [Tile.BLUE]

    game.state.center.append(Tile.BLUE)

    moves = [agent.choose_move(game) for _ in range(100)]

    # Among blue moves, majority should go to row 2
    blue_moves = [m for m in moves if m.color == Tile.BLUE and m.destination >= 0]
    if blue_moves:
        partial_picks = sum(1 for m in blue_moves if m.destination == 2)
        assert partial_picks / len(blue_moves) > 0.5, (
            "Expected >50% of blue moves to target row 2, "
            f"got {partial_picks}/{len(blue_moves)}"
        )

    # But BLUE should not dominate color selection overall
    non_floor = [m for m in moves if m.destination != -2]
    blue_non_floor = [m for m in non_floor if m.color == Tile.BLUE]
    if non_floor:
        assert (
            len(blue_non_floor) / len(non_floor) < 0.9
        ), "Color selection appears biased toward BLUE"
