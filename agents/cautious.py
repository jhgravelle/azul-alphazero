# agents/cautious.py
"""An agent that avoids placing tiles on the floor when possible."""

import random

from agents.base import Agent
from agents.move_filters import non_floor_moves
from engine.game import Game, Move


class CautiousAgent(Agent):
    """Selects a random move from those that don't target the floor.

    Falls back to including floor moves if every legal move targets the
    floor (relies on non_floor_moves to handle the fallback).
    """

    def choose_move(self, game: Game) -> Move:
        return random.choice(non_floor_moves(game.legal_moves()))

    def policy_distribution(self, game: Game) -> list[tuple[Move, float]]:
        candidates = non_floor_moves(game.legal_moves())
        prob = 1.0 / len(candidates)
        return [(m, prob) for m in candidates]
