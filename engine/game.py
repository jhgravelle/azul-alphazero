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
        """Return a fast independent copy of this game.

        Bypasses __init__ and GameState.__post_init__ entirely — no fresh
        bag shuffle, no empty board construction. All mutable state is
        copied with direct list operations.
        """
        g = object.__new__(Game)
        s = object.__new__(GameState)

        s.current_player = self.state.current_player
        s.round = self.state.round
        s.players = [p.clone() for p in self.state.players]
        s.factories = [f[:] for f in self.state.factories]
        s.center = self.state.center[:]
        s.bag = self.state.bag[:]
        s.discard = self.state.discard[:]

        g.state = s
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

    def advance_round_if_needed(self) -> None:
        """Call setup_round if the round just ended and the game is not over.

        Simulation loops must call this after every make_move to keep the game
        state valid. The API calls this too, but may substitute factory setup
        mode instead.
        """
        if self.is_game_over():
            return
        sources_empty = (
            all(len(f) == 0 for f in self.state.factories)
            and len(self.state.center) == 0
        )
        if sources_empty:
            self.setup_round()

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

    def _take_from_source(self, player: Board, move: Move) -> list[Tile]:
        """Remove tiles from the source, send leftovers to center.

        Returns the chosen tiles. Moves the first-player marker to the
        player's floor line if it is in the source.
        """
        source = (
            self.state.center
            if move.source == CENTER
            else self.state.factories[move.source]
        )
        chosen = [t for t in source if t == move.tile]
        leftover = [t for t in source if t != move.tile and t != Tile.FIRST_PLAYER]
        if Tile.FIRST_PLAYER in source:
            player.floor_line.append(Tile.FIRST_PLAYER)
        source.clear()
        self.state.center.extend(leftover)
        return chosen

    def _place_tiles(self, player: Board, move: Move, chosen: list[Tile]) -> None:
        """Place chosen tiles on the destination pattern line or floor.

        Overflow tiles beyond the line's capacity go to the floor.
        """
        if move.destination == FLOOR:
            player.floor_line.extend(chosen)
        else:
            line = player.pattern_lines[move.destination]
            capacity = move.destination + 1
            space = capacity - len(line)
            line.extend(chosen[:space])
            player.floor_line.extend(chosen[space:])

    def _end_turn(self) -> None:
        """Advance to the next player and trigger end-of-round scoring if all
        sources are empty. Round setup is left to the caller."""
        self.state.current_player = (self.state.current_player + 1) % len(
            self.state.players
        )
        if (
            all(len(f) == 0 for f in self.state.factories)
            and len(self.state.center) == 0
        ):
            self.score_round()
            if self.is_game_over():
                self.score_game()

    def make_move(self, move: Move) -> None:
        """Apply a move to the current game state."""
        player = self.state.players[self.state.current_player]
        chosen = self._take_from_source(player, move)
        self._place_tiles(player, move, chosen)
        self._end_turn()

    # ------------------------------------------------------------------
    # End-of-round scoring
    # ------------------------------------------------------------------

    def _score_floor(self, player: Board) -> None:
        """Apply floor line penalties to the player and clear the floor."""
        player.score = max(0, player.score + score_floor_penalty(player.floor_line))
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
