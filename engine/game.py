# engine/game.py

"""Core game controller for Azul."""

from dataclasses import dataclass
from engine.board import Board
from engine.constants import (
    Tile,
    BOARD_SIZE,
    COLOR_TILES,
    TILES_PER_FACTORY,
    COLUMN_FOR_TILE_IN_ROW,
)
from engine.game_state import GameState
from engine.scoring import score_floor_penalty, score_placement, score_wall_bonus
import random
import logging

logger = logging.getLogger(__name__)

CENTER = -1
FLOOR = -2


@dataclass
class Move:
    source: int
    tile: Tile
    destination: int


class Game:
    """The main game controller for Azul.

    Manages game state, enforces rules, and exposes methods for players to
    take actions. Mutation lives here; pure scoring logic lives in scoring.py.
    """

    def __init__(self):
        """Initialize a new game with a fresh GameState."""
        self.state = GameState()

    def clone(self) -> "Game":
        """Return a fast independent copy of this game."""
        g = object.__new__(Game)
        g.state = self.state.clone()
        return g

    # ------------------------------------------------------------------
    # Round setup
    # ------------------------------------------------------------------

    def _refill_bag(self) -> None:
        """Refill the bag from the discard pile and shuffle it."""
        if not self.state.discard:
            return
        self.state.bag.extend(self.state.discard)
        self.state.discard.clear()
        random.shuffle(self.state.bag)
        logger.debug("refilled bag from discard and shuffled")

    def setup_round(self, factories: list[list[Tile]] | None = None) -> None:
        """Set up the factories and center for a new round.

        If factories is provided, those tile lists are loaded directly and the
        bag is not touched. If None, factories are filled by drawing randomly
        from the bag as normal.
        """
        self.state.round += 1
        self.state.center.clear()
        self.state.center.append(Tile.FIRST_PLAYER)
        if factories is not None:
            for factory, tiles in zip(self.state.factories, factories):
                factory.clear()
                factory.extend(tiles)
        else:
            for factory in self.state.factories:
                factory.clear()
                for _ in range(TILES_PER_FACTORY):
                    if not self.state.bag:
                        self._refill_bag()
                    if not self.state.bag:
                        logger.debug("no tiles remaining to fill factories")
                        return
                    factory.append(self.state.bag.pop())

    def is_round_over(self) -> bool:
        """Return True when no color tiles remain in any factory or the center."""
        factories_empty = all(len(f) == 0 for f in self.state.factories)
        center_empty = not any(t in COLOR_TILES for t in self.state.center)
        return factories_empty and center_empty

    def advance(self, *, skip_setup: bool = False) -> bool:
        """Advance past the current move: rotate player, then handle phase
        transitions if the round or game just ended.

        skip_setup: if True, setup_round() is skipped after scoring. The
        caller is responsible for setting up the next round (e.g. manual
        factory setup or replaying recorded factories). Default False.

        Returns True if a round boundary was crossed (round was scored),
        False otherwise.
        """
        self.next_player()
        if self.is_round_over():
            self.score_round()
            if self.is_game_over():
                self.score_game()
            elif not skip_setup:
                self.setup_round()
            return True
        return False

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def _is_valid_destination(
        self, player: Board, tile: Tile, destination: int
    ) -> bool:
        """Return True if tile can be placed on the destination pattern line."""
        if destination == FLOOR:
            return True
        line = player.pattern_lines[destination]
        if len(line) == destination + 1:
            return False
        if tile in player.wall[destination]:
            return False
        if not line:
            return True
        return line[0] == tile

    def legal_moves(self) -> list[Move]:
        """Return a list of legal moves for the current player."""
        moves = []
        player = self.state.players[self.state.current_player]
        sources = list(enumerate(self.state.factories))
        sources.append((CENTER, self.state.center))
        for source_index, source_tiles in sources:
            tile_options = set(t for t in source_tiles if t in COLOR_TILES)
            for tile in tile_options:
                for destination in [*range(BOARD_SIZE), FLOOR]:
                    if self._is_valid_destination(player, tile, destination):
                        moves.append(
                            Move(
                                source=source_index,
                                tile=tile,
                                destination=destination,
                            )
                        )
        return moves

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def _take_from_source(self, move: Move) -> list[Tile]:
        """Remove chosen color tiles and FIRST_PLAYER from the source.

        Sends non-chosen color tiles to the center. Returns the chosen
        color tiles plus FIRST_PLAYER if it was present in the source.
        """
        source = (
            self.state.center
            if move.source == CENTER
            else self.state.factories[move.source]
        )
        chosen = [t for t in source if t == move.tile or t == Tile.FIRST_PLAYER]
        leftover = [t for t in source if t in COLOR_TILES and t != move.tile]
        source.clear()
        self.state.center.extend(leftover)
        return chosen

    def _place_tiles(self, player: Board, move: Move, chosen: list[Tile]) -> None:
        """Place chosen tiles on the destination pattern line or floor.

        FIRST_PLAYER tiles go directly to the floor. color tiles fill the
        pattern line up to capacity; overflow goes to the floor.
        """
        color_tiles = [t for t in chosen if t != Tile.FIRST_PLAYER]
        if Tile.FIRST_PLAYER in chosen:
            player.floor_line.append(Tile.FIRST_PLAYER)
        if move.destination == FLOOR:
            player.floor_line.extend(color_tiles)
        else:
            line = player.pattern_lines[move.destination]
            capacity = move.destination + 1
            space = capacity - len(line)
            line.extend(color_tiles[:space])
            player.floor_line.extend(color_tiles[space:])

    def next_player(self) -> None:
        """Advance current_player to the next player in turn order."""
        self.state.current_player = (self.state.current_player + 1) % len(
            self.state.players
        )

    def make_move(self, move: Move) -> None:
        """Apply a move to the current game state."""
        player = self.state.players[self.state.current_player]
        chosen = self._take_from_source(move)
        self._place_tiles(player, move, chosen)

    # ------------------------------------------------------------------
    # End-of-round scoring
    # ------------------------------------------------------------------

    def _score_floor(self, player: Board) -> None:
        """Apply floor line penalties to the player and clear the floor."""
        penalty = score_floor_penalty(player.floor_line)
        player.clamped_points += max(0, -(player.score + penalty))
        player.score = max(0, player.score + penalty)
        self.state.discard.extend(
            t for t in player.floor_line if t != Tile.FIRST_PLAYER
        )
        player.floor_line.clear()

    def score_round(self) -> None:
        """Score the end of a round for all players.

        For each player: moves completed pattern line tiles to the wall and
        scores them, discards extras, leaves incomplete lines alone, then
        applies floor penalties. The player who held the first-player marker
        becomes the starting player next round.
        """
        for player in self.state.players:
            for row, line in enumerate(player.pattern_lines):
                if len(line) < row + 1:
                    continue
                tile = line[0]
                column = COLUMN_FOR_TILE_IN_ROW[tile][row]
                player.wall[row][column] = tile
                player.score += score_placement(player.wall, row, column)
                self.state.discard.extend(line[1:])
                player.pattern_lines[row] = []

        for i, player in enumerate(self.state.players):
            if Tile.FIRST_PLAYER in player.floor_line:
                self.state.current_player = i
                break

        for player in self.state.players:
            self._score_floor(player)

    # ------------------------------------------------------------------
    # Game-over detection and end-of-game scoring
    # ------------------------------------------------------------------

    def is_game_over(self) -> bool:
        """Return True if any player has completed at least one wall row."""
        for player in self.state.players:
            if any(all(cell is not None for cell in row) for row in player.wall):
                return True
        return False

    def score_game(self) -> None:
        """Apply end-of-game bonus scoring to all players."""
        for player in self.state.players:
            player.score += score_wall_bonus(player.wall)
