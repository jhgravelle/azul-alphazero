# engine / game.py

# """Core game state dataclasses for Azul."""

# import logging
# from dataclasses import dataclass, field
from enum import Enum, auto

# logger = logging.getLogger(__name__)


class Tile(Enum):
    """The five tile colors in Azul, plus the first-player marker."""

    BLUE = auto()
    YELLOW = auto()
    RED = auto()
    BLACK = auto()
    WHITE = auto()
    FIRST_PLAYER = auto()


# @dataclass
# class PlayerBoard:
#     """Represents a single player's board.

#     Attributes:
#         score: The player's current score.
#         pattern_lines: 5 rows; row i can hold at most i+1 tiles of one color.
#         wall: 5x5 grid of placed tiles (None = empty).
#         floor_line: Tiles dropped here incur minus-point penalties.
#     """

#     score: int = 0
#     pattern_lines: list[list[Tile]] = field(
#         default_factory=lambda: [[] for _ in range(5)]
#     )
#     wall: list[list[Tile | None]] = field(
#         default_factory=lambda: [[None] * 5 for _ in range(5)]
#     )
#     floor_line: list[Tile] = field(default_factory=list)


# @dataclass
# class GameState:
#     """The complete state of an Azul game.

#     Attributes:
#         num_players: 2, 3, or 4.
#         players: One PlayerBoard per player.
#         factories: The factory displays (each will hold 4 tiles when filled).
#         centre: The centre pool where leftover tiles and the first-player
#                 marker are placed.
#         current_player: Index into players — whose turn it is.
#     """

#     num_players: int = 2
#     players: list[PlayerBoard] = field(default_factory=list)
#     factories: list[list[Tile]] = field(default_factory=list)
#     centre: list[Tile] = field(default_factory=list)
#     current_player: int = 0

#     def __post_init__(self) -> None:
#         """Initialise players and factories based on num_players."""
#         if not self.players:
#             self.players = [PlayerBoard() for _ in range(self.num_players)]
#             logger.debug("Created %d player boards", self.num_players)

#         if not self.factories:
#             num_factories = 2 * self.num_players + 1
#             self.factories = [[] for _ in range(num_factories)]
#             logger.debug("Created %d factory displays", num_factories)
