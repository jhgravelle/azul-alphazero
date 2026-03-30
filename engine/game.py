# engine / game.py

"""Core game controller for Azul."""

from dataclasses import dataclass
from engine.constants import BOARD_SIZE, TILES_PER_FACTORY
from engine.game_state import GameState, PlayerBoard, Tile
import random
import logging

logger = logging.getLogger(__name__)

CENTER = -1
FLOOR = -1


@dataclass
class Move:
    source: int
    color: Tile
    destination: int


class Game:
    """The main game controller for Azul.

    This class is responsible for managing the game state, enforcing rules, and
    providing methods for players to take actions. It serves as the central
    point of interaction for the game logic.
    """

    def __init__(self):
        """Initialize a new game with a fresh GameState."""
        self.state = GameState()

    def _refill_bag(self) -> None:
        """Refill the bag from the discard pile and shuffle it."""
        if not self.state.discard:
            return
        self.state.bag.extend(self.state.discard)
        self.state.discard.clear()
        random.shuffle(self.state.bag)
        logger.debug("refilled bag from discard and shuffled")

    def setup_round(self) -> None:
        """Set up the factories and center for a new round."""
        for factory in self.state.factories:
            factory.clear()  # safety check
            for _ in range(TILES_PER_FACTORY):
                if not self.state.bag:
                    self._refill_bag()
                if not self.state.bag:
                    logger.debug("no tiles remaining to fill factories")
                    return
                factory.append(self.state.bag.pop())

    def _is_valid_destination(
        self, player: PlayerBoard, color: Tile, destination: int
    ) -> bool:
        """Return True if the color can be placed on the destination pattern line."""
        if destination == FLOOR:
            return True
        line = player.pattern_lines[destination]
        if len(line) == destination + 1:  # row i has capacity i+1
            return False
        if color in player.wall[destination]:
            return False
        if not line:
            return True
        return line[0] == color

    def legal_moves(self) -> list[Move]:
        """Return a list of legal moves for the current player."""
        moves = []
        player = self.state.players[self.state.current_player]
        sources = list(enumerate(self.state.factories))
        sources.append((CENTER, self.state.center))
        for source_index, source_tiles in sources:
            colors = set(t for t in source_tiles if t != Tile.FIRST_PLAYER)
            for color in colors:
                for destination in range(BOARD_SIZE):
                    if self._is_valid_destination(player, color, destination):
                        moves.append(
                            Move(
                                source=source_index,
                                color=color,
                                destination=destination,
                            )
                        )
                moves.append(Move(source=source_index, color=color, destination=FLOOR))
        return moves
