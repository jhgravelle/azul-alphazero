# agents/move_filters.py

"""Shared move filtering helpers for agents.

These are pure functions that take a list of moves and return a filtered
list. They never modify the game state.
"""

from engine.game import Move, FLOOR


def non_floor_moves(moves: list[Move]) -> list[Move]:
    """Return moves that don't target the floor, with fallback.

    If all moves target the floor, returns the original list so the
    caller always has at least one move to choose from.
    """
    filtered = [m for m in moves if m.destination != FLOOR]
    return filtered if filtered else moves
