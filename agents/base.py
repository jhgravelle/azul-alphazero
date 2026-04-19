# agents/base.py
"""Abstract base class for all Azul agents."""

from abc import ABC, abstractmethod
from engine.game import Game, Move


class Agent(ABC):
    """Abstract base class that every Azul agent must inherit from.
    To create a new agent, subclass this and implement choose_move.
    """

    @abstractmethod
    def choose_move(self, game: Game) -> Move:
        """Select and return a legal move for the current game state.
        Args:
            game: The current Game instance. Do not modify it — read only.
        Returns:
            A Move object that is legal in the current game state.
        """

    def policy_distribution(self, game: Game) -> list[tuple[Move, float]]:
        """Return the distribution over legal moves this agent samples from.

        The default implementation returns a uniform distribution over all
        legal moves, matching RandomAgent's behavior. Subclasses that apply
        heuristic filters should override this to return the filtered
        distribution they actually sample from.

        Args:
            game: The current Game instance. Do not modify it — read only.

        Returns:
            A list of (move, probability) tuples. Probabilities sum to 1.0
            across the returned moves. Order is not guaranteed to match
            game.legal_moves().
        """
        legal = game.legal_moves()
        prob = 1.0 / len(legal)
        return [(m, prob) for m in legal]
