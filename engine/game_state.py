# engine / game.py

"""Core game state dataclasses for Azul."""

from dataclasses import dataclass, field
from engine.board import Board
from engine.tile import Tile
from engine.constants import PLAYERS, TILES_PER_COLOR
import random
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

    def __post_init__(self) -> None:
        if not self.players:
            self.players = [Board() for _ in range(PLAYERS)]
            logger.debug(f"Created {PLAYERS} player boards")

        if not self.factories:
            num_factories = 2 * PLAYERS + 1
            self.factories = [[] for _ in range(num_factories)]
            logger.debug(f"Created {num_factories} factories")

        if not self.bag:
            self.bag = [
                color for color in Tile if color != Tile.FIRST_PLAYER
            ] * TILES_PER_COLOR
            logger.debug(f"Created bag with {len(self.bag)} tiles")
            random.shuffle(self.bag)  # Shuffle the bag at the start of the game
