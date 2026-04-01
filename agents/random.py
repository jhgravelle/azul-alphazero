# agents/random.py

"""An agent that selects a random legal move, preferring non-floor moves."""

import random

from agents.base import Agent
from engine.game import Game, Move


class RandomAgent(Agent):
    """Selects a random legal move, avoiding the floor if possible."""

    def choose_move(self, game: Game) -> Move:
        """Return a random legal move, preferring non-floor destinations.

        Args:
            game: The current Game instance. Read only.

        Returns:
            A randomly selected Move, from non-floor moves if any exist.
        """
        moves = game.legal_moves()
        non_floor = [m for m in moves if m.destination != -2]
        return random.choice(non_floor if non_floor else moves)
