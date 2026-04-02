# agents/random.py

"""An agent that selects a true uniformly random legal move."""

import random

from agents.base import Agent
from engine.game import Game, Move


class RandomAgent(Agent):
    """Selects a legal move uniformly at random with no heuristics.

    This is the standard baseline opponent for benchmarking other agents.
    """

    def choose_move(self, game: Game) -> Move:
        """Return a uniformly random legal move.

        Args:
            game: The current Game instance. Read only.

        Returns:
            A randomly selected Move from all legal moves.
        """
        return random.choice(game.legal_moves())
