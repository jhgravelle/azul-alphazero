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
