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
        """Return a heuristically filtered random move."""
        candidates = non_floor_moves(game.legal_moves())
        chosen_color = random.choice(list({m.tile for m in candidates}))
        color_moves = self._color_candidates(game, candidates, chosen_color)
        return random.choice(color_moves)

    def policy_distribution(self, game: Game) -> list[tuple[Move, float]]:
        candidates = non_floor_moves(game.legal_moves())
        colors = list({m.tile for m in candidates})
        num_colors = len(colors)
        result: list[tuple[Move, float]] = []
        for color in colors:
            color_moves = self._color_candidates(game, candidates, color)
            prob_within_color = 1.0 / len(color_moves)
            prob_overall = (1.0 / num_colors) * prob_within_color
            for m in color_moves:
                result.append((m, prob_overall))
        return result

    @staticmethod
    def _color_candidates(game: Game, candidates: list[Move], color) -> list[Move]:
        """Return the moves this agent would sample from for a given color."""
        player = game.state.players[game.state.current_player]
        color_moves = [m for m in candidates if m.tile == color]
        preferred = [
            m
            for m in color_moves
            if m.destination >= 0 and len(player.pattern_lines[m.destination]) > 0
        ]
        return preferred if preferred else color_moves
