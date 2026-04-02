# agents/cautious.py

"""An agent that avoids the floor line when possible."""

import random

from agents.base import Agent
from engine.game import Game, Move


class CautiousAgent(Agent):
    """Selects a random legal move, but avoids sending tiles to the floor
    if any pattern line move is available.

    Heuristic: avoid the floor if possible, otherwise pick uniformly at random.
    """

    def choose_move(self, game: Game) -> Move:
        """Return a random move, preferring non-floor destinations.

        Args:
            game: The current Game instance. Read only.

        Returns:
            A randomly selected Move from non-floor moves if any exist,
            otherwise any legal move.
        """
        moves = game.legal_moves()
        non_floor = [m for m in moves if m.destination != -2]
        return random.choice(non_floor if non_floor else moves)
