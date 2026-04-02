# agents/efficient.py

"""An agent that prefers placing tiles on partially filled pattern lines."""

import random

from agents.base import Agent
from engine.game import Game, Move


class EfficientAgent(Agent):
    """Selects a random legal move, but prefers placing tiles on a pattern
    line that already has tiles of the same color.

    Heuristic: among all legal moves, prefer completing lines already in
    progress. Will still send tiles to the floor if that's the only option.
    """

    def choose_move(self, game: Game) -> Move:
        """Return a random move, preferring partial line destinations.

        Args:
            game: The current Game instance. Read only.

        Returns:
            A randomly selected Move from partial-line moves if any exist,
            otherwise any legal move.
        """
        moves = game.legal_moves()
        player = game.state.players[game.state.current_player]
        preferred = [
            m
            for m in moves
            if m.destination >= 0 and len(player.pattern_lines[m.destination]) > 0
        ]
        return random.choice(preferred if preferred else moves)
