# agents/random.py

"""An agent that selects a random legal move with basic heuristics."""

import random

from agents.base import Agent
from engine.game import Game, Move


class RandomAgent(Agent):
    """Selects a random legal move with two simple heuristics:
    1. Avoids the floor if any pattern line move is available.
    2. Once a color is chosen, prefers placing on a partial line of that color.
    """

    def choose_move(self, game: Game) -> Move:
        """Return a heuristically filtered random move.

        Args:
            game: The current Game instance. Read only.

        Returns:
            A randomly selected Move from the filtered candidate list.
        """
        moves = game.legal_moves()

        # Heuristic 1 — avoid the floor if possible
        non_floor = [m for m in moves if m.destination != -2]
        candidates = non_floor if non_floor else moves

        # Pick a random color from the available candidates
        available_colors = list({m.color for m in candidates})
        chosen_color = random.choice(available_colors)
        color_moves = [m for m in candidates if m.color == chosen_color]

        # Heuristic 2 — among moves for that color, prefer a partial line
        player = game.state.players[game.state.current_player]
        preferred = [
            m
            for m in color_moves
            if m.destination >= 0 and len(player.pattern_lines[m.destination]) > 0
        ]

        return random.choice(preferred if preferred else color_moves)
