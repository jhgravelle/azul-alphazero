# agents/greedy.py

"""An agent combining floor-avoidance and partial-line preference heuristics."""

import random

from agents.base import Agent
from agents.move_filters import non_floor_moves
from engine.game import Game, Move


class GreedyAgent(Agent):
    """Selects moves using two heuristics applied in sequence:

    1. Avoid the floor if any pattern line move is available.
    2. Once a color is chosen, prefer placing on a partial line of that color.

    This is the strongest of the heuristic agents and the recommended
    default opponent for human players.
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
        candidates = non_floor_moves(moves)

        # Pick a random color from the available candidates
        available_colors = list({m.tile for m in candidates})
        chosen_color = random.choice(available_colors)
        color_moves = [m for m in candidates if m.tile == chosen_color]

        # Heuristic 2 — among moves for that color, prefer a partial line
        player = game.state.players[game.state.current_player]
        preferred = [
            m
            for m in color_moves
            if m.destination >= 0 and len(player.pattern_lines[m.destination]) > 0
        ]
        return random.choice(preferred if preferred else color_moves)
