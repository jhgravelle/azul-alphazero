# tests/test_move_filters.py

"""Tests for shared move filtering helpers."""

from engine.game import Game, Move, FLOOR
from agents.move_filters import non_floor_moves


def test_non_floor_moves_excludes_floor():
    game = Game()
    game.setup_round()
    moves = game.legal_moves()
    filtered = non_floor_moves(moves)
    assert all(m.destination != FLOOR for m in filtered)


def test_non_floor_moves_returns_non_empty_when_pattern_lines_available():
    game = Game()
    game.setup_round()
    filtered = non_floor_moves(game.legal_moves())
    assert len(filtered) > 0


def test_non_floor_moves_falls_back_to_all_when_only_floor():
    """When every legal move targets the floor, return all moves."""
    from engine.constants import Tile

    floor_only = [
        Move(source=0, tile=Tile.BLUE, destination=FLOOR),
        Move(source=1, tile=Tile.RED, destination=FLOOR),
    ]
    result = non_floor_moves(floor_only)
    assert result == floor_only
