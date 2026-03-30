# engine / game.py

"""Core game state dataclasses for Azul."""

from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

BOARD_SIZE = 5  # also equal to the number of tile colors
PLAYERS = 2  # number of players in a new GameState
TILES_PER_COLOR = (
    20  # number of tiles of each color in the bag at the start of the game
)


class Tile(Enum):
    """The five tile colors in Azul, plus the first-player marker."""

    BLUE = auto()
    YELLOW = auto()
    RED = auto()
    BLACK = auto()
    WHITE = auto()
    FIRST_PLAYER = auto()


@dataclass
class PlayerBoard:
    """Represents a single player's board.

    Attributes:
        score: The player's current score.
        pattern_lines: 5 rows; row i can hold at most i+1 tiles of one color.
        wall: 5x5 grid of placed tiles (None = empty).
        floor_line: Tiles dropped here incur minus-point penalties.
    """

    score: int = 0

    pattern_lines: list[list[Tile]] = field(
        default_factory=lambda: [[] for _ in range(BOARD_SIZE)]
    )

    wall: list[list[Tile | None]] = field(
        default_factory=lambda: [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    )

    floor_line: list[Tile] = field(default_factory=list)


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

    players: list[PlayerBoard] = field(default_factory=list)
    current_player: int = 0
    factories: list[list[Tile]] = field(default_factory=list)
    center: list[Tile] = field(default_factory=list)
    bag: list[Tile] = field(default_factory=list)
    discard: list[Tile] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.players:
            self.players = [PlayerBoard() for _ in range(PLAYERS)]
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
