# agents/random.py

"""A agent that selects a random legal move."""

import random

from agents.base import Agent
from engine.game import Game, Move


class RandomAgent(Agent):
    """Selects a uniformly random legal move each turn."""

    def choose_move(self, game: Game) -> Move:
        """Return a random legal move for the current game state.

        Args:
            game: The current Game instance. Read only.

        Returns:
            A randomly selected Move from the list of legal moves.
        """
        return random.choice(game.legal_moves())
