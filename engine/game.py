# engine / game.py

"""Core game controller for Azul."""

from dataclasses import dataclass
from engine.board import Board
from engine.tile import Tile, COLORS
from engine.constants import BOARD_SIZE, TILES_PER_FACTORY, FLOOR_PENALTIES
from engine.game_state import GameState
import random
import logging

logger = logging.getLogger(__name__)

CENTER = -1
FLOOR = -2
_WALL_SEQUENCE = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]

WALL_PATTERN: list[list[Tile]] = [
    [_WALL_SEQUENCE[(col - row) % BOARD_SIZE] for col in range(BOARD_SIZE)]
    for row in range(BOARD_SIZE)
]


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
        self.state.center.clear()
        self.state.center.append(Tile.FIRST_PLAYER)
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
        self, player: Board, color: Tile, destination: int
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

    def wall_column_for(self, *, row: int, color: Tile) -> int:
        """Return the column index where a color tile belongs in a given wall row."""
        return WALL_PATTERN[row].index(color)

    def make_move(self, move: Move) -> None:
        """Apply a move to the current game state.

        Trusts that the move is legal. Removes tiles from the source,
        sends leftovers to the center, places chosen tiles on the
        destination pattern line or floor, and advances the current player.
        """
        player = self.state.players[self.state.current_player]

        # 1. Pull tiles from the source
        if move.source == CENTER:
            source = self.state.center
        else:
            source = self.state.factories[move.source]

        chosen = [t for t in source if t == move.color]
        leftover = [t for t in source if t != move.color and t != Tile.FIRST_PLAYER]

        # 2. Handle first-player marker
        if Tile.FIRST_PLAYER in source:
            player.floor_line.append(Tile.FIRST_PLAYER)

        # 3. Clear the source and place the leftovers in the center
        source.clear()
        self.state.center.extend(leftover)

        # 4. Place chosen tiles on destination
        if move.destination == FLOOR:
            player.floor_line.extend(chosen)
        else:
            line = player.pattern_lines[move.destination]
            capacity = move.destination + 1
            space = capacity - len(line)
            line.extend(chosen[:space])
            player.floor_line.extend(chosen[space:])

        # 6. Advance to next player
        self.state.current_player = (self.state.current_player + 1) % len(
            self.state.players
        )

        # 7. Check if the round is over
        factories_empty = all(len(f) == 0 for f in self.state.factories)
        center_empty = len(self.state.center) == 0
        if factories_empty and center_empty:
            self.score_round()
            if not self.is_game_over():
                self.setup_round()

    def _score_placement(
        self, wall: list[list[Tile | None]], row: int, col: int
    ) -> int:
        """Score a single tile just placed at (row, col) on the wall.

        Counts the horizontal and vertical runs through that cell.
        A lone tile (no neighbours) scores 1.
        """

        def run_length(dr: int, dc: int) -> int:
            """Count tiles in one direction from (row, col), excluding the origin."""
            length = 0
            r, c = row + dr, col + dc
            while (
                0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and wall[r][c] is not None
            ):
                length += 1
                r += dr
                c += dc
            return length

        h = run_length(0, -1) + 1 + run_length(0, 1)  # left + self + right
        v = run_length(-1, 0) + 1 + run_length(1, 0)  # up + self + down

        if h == 1 and v == 1:
            return 1  # no neighbours — lone tile
        score = 0
        if h > 1:
            score += h
        if v > 1:
            score += v
        return score

    def _score_floor(self, player: "Board") -> None:
        """Apply floor line penalties to the player and clear the floor."""
        penalty = sum(
            FLOOR_PENALTIES[i]
            for i in range(min(len(player.floor_line), len(FLOOR_PENALTIES)))
        )
        player.score = max(0, player.score + penalty)
        self.state.discard.extend(player.floor_line)
        player.floor_line.clear()

    def score_round(self) -> None:
        """Score the end of a round for all players.

        For each player:
        - Full pattern lines: move one tile to the wall, score it, discard the rest
        - Incomplete pattern lines: leave them alone
        - Apply floor penalties and clear the floor
        After scoring, the player who held the first player marker starts next round.
        """
        for player in self.state.players:
            for row, line in enumerate(player.pattern_lines):
                capacity = row + 1
                if len(line) < capacity:
                    continue

                color = line[0]
                col = self.wall_column_for(row=row, color=color)
                player.wall[row][col] = color
                player.score += self._score_placement(player.wall, row, col)
                self.state.discard.extend(line[1:])
                player.pattern_lines[row] = []

        # Find who has the first player marker before floors are cleared
        for i, player in enumerate(self.state.players):
            if Tile.FIRST_PLAYER in player.floor_line:
                self.state.current_player = i
                break

        for player in self.state.players:
            self._score_floor(player)

    def is_game_over(self) -> bool:
        """Return True if any player has completed at least one wall row."""
        for player in self.state.players:
            for row in player.wall:
                if all(cell is not None for cell in row):
                    return True
        return False

    def score_game(self) -> None:
        """Apply end-of-game bonus scoring to all players.

        Awards:
        - 2 points per complete horizontal wall row
        - 7 points per complete vertical wall column
        - 10 points per color with all 5 tiles placed on the wall
        """
        for player in self.state.players:
            # Complete rows — 2 points each
            for row in player.wall:
                if all(cell is not None for cell in row):
                    player.score += 2

            # Complete columns — 7 points each
            for col in range(BOARD_SIZE):
                if all(player.wall[row][col] is not None for row in range(BOARD_SIZE)):
                    player.score += 7

            # Complete colors — 10 points each
            for color in COLORS:
                if all(
                    player.wall[row][self.wall_column_for(row=row, color=color)]
                    == color
                    for row in range(BOARD_SIZE)
                ):
                    player.score += 10
