# engine / game.py

"""Core game state dataclasses for Azul."""

from dataclasses import dataclass, field
from engine.board import Board
from engine.constants import Tile, COLOR_TILES, PLAYERS, TILES_PER_COLOR
import logging

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """The complete state of an Azul game.

    Attributes:
        PLAYERS currently hardcoded to two via module-level constant.
        players: One PlayerBoard per player.
        factories: The factory displays (each will hold 4 tiles when filled).
        center: The center pool where leftover tiles and the first-player
                marker are placed.
        current_player: Index into players — whose turn it is.
    """

    players: list[Board] = field(default_factory=list)
    current_player: int = 0
    factories: list[list[Tile]] = field(default_factory=list)
    center: list[Tile] = field(default_factory=list)
    bag: list[Tile] = field(default_factory=list)
    discard: list[Tile] = field(default_factory=list)
    round: int = 0

    def __post_init__(self) -> None:
        if not self.players:
            self.players = [Board() for _ in range(PLAYERS)]
            logger.debug(f"Created {PLAYERS} player boards")

        if not self.factories:
            num_factories = 2 * PLAYERS + 1
            self.factories = [[] for _ in range(num_factories)]
            logger.debug(f"Created {num_factories} factories")

        if not self.bag:
            self.bag = [color for color in COLOR_TILES] * TILES_PER_COLOR
            logger.debug(f"Created bag with {len(self.bag)} tiles")

    def clone(self) -> "GameState":
        """Return a fast independent copy of this game state.

        Bypasses __post_init__ entirely — no fresh bag shuffle, no empty
        board construction. All mutable state is copied with direct list
        operations.
        """
        s = object.__new__(GameState)
        s.current_player = self.current_player
        s.round = self.round
        s.players = [p.clone() for p in self.players]
        s.factories = [f[:] for f in self.factories]
        s.center = self.center[:]
        s.bag = self.bag[:]
        s.discard = self.discard[:]
        return s
