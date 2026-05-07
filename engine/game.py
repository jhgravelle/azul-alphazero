# engine/game.py
"""Core game controller for Azul."""

import logging
import random
from dataclasses import dataclass

from engine.constants import (
    BOARD_SIZE,
    CENTER,
    CHAR_TILE,
    COLOR_TILES,
    FLOOR,
    MOVE_DEST_FLOOR,
    MOVE_MARKER_FIRST_PLAYER,
    MOVE_MARKER_NORMAL,
    MOVE_MARKER_UNKNOWN,
    MOVE_SOURCE_CENTER,
    NUMBER_OF_FACTORIES,
    PLAYERS,
    TILE_CHAR,
    TILES_PER_COLOR,
    TILES_PER_FACTORY,
    Tile,
)
from engine.player import Player

logger = logging.getLogger(__name__)


@dataclass
class Move:
    """A single game action: take all tiles of one color from a source and
    place them on a destination pattern line or the floor.

    Attributes:
        source:      Factory index (0..N-1) or CENTER (-1).
        tile:        The tile color being taken.
        destination: Pattern line index (0..4) or FLOOR (-2).
        count:       Number of color tiles taken (excludes FIRST_PLAYER).
        took_first:  True if FIRST_PLAYER was also taken from the center.

    Compact string format: {count}{tile}{marker}{source}{destination}
        count:       number of color tiles
        tile:        single char from TILE_CHAR
        marker:      - normally, + if FIRST_PLAYER was taken
        source:      C for center, 1-5 for factory (1-based)
        destination: F for floor, 1-5 for pattern line row (1-based)

    Examples:
        2W-C3 — 2 white from center to row 3
        2W+C3 — same but also took the first-player tile
        1B-2F — 1 blue from factory 2 to floor
        4R+1F — 4 red from factory 1 to floor, took first-player tile
    """

    tile: Tile
    source: int
    destination: int
    count: int = 0
    took_first: bool = False

    def __str__(self) -> str:
        marker = (
            MOVE_MARKER_UNKNOWN
            if self.count == 0
            else MOVE_MARKER_FIRST_PLAYER if self.took_first else MOVE_MARKER_NORMAL
        )
        source_str = (
            MOVE_SOURCE_CENTER if self.source == CENTER else str(self.source + 1)
        )
        dest_str = (
            MOVE_DEST_FLOOR if self.destination == FLOOR else str(self.destination + 1)
        )
        return f"{self.count}{TILE_CHAR[self.tile]}{marker}{source_str}{dest_str}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return NotImplemented
        return (
            self.tile == other.tile
            and self.source == other.source
            and self.destination == other.destination
        )

    def __hash__(self) -> int:
        return hash((self.tile, self.source, self.destination))

    @classmethod
    def from_str(cls, s: str) -> "Move":
        """Parse a compact move string back into a Move.

        Format: {count}{tile}{marker}{source}{destination}
        count is 1 digit normally, 2 digits when >= 10.

        Raises ValueError if the string is not a valid move format.

        Examples:
            Move.from_str("2W-C3")  # 2 white, center, row 3, no first player
            Move.from_str("4R+1F")  # 4 red, factory 1, floor, took first player
            Move.from_str("10R+CF") # 10 red, center, floor, took first player
        """
        try:
            count = int(s[:-4])
            tile = CHAR_TILE[s[-4]]
            if tile is None:
                raise ValueError(f"invalid tile char: {s[-4]!r}")
            took_first = s[-3] == MOVE_MARKER_FIRST_PLAYER
            source = CENTER if s[-2] == MOVE_SOURCE_CENTER else int(s[-2]) - 1
            destination = FLOOR if s[-1] == MOVE_DEST_FLOOR else int(s[-1]) - 1
            return cls(
                tile=tile,
                source=source,
                destination=destination,
                count=count,
                took_first=took_first,
            )
        except (IndexError, KeyError) as exc:
            raise ValueError(f"invalid move string: {s!r}") from exc


class Game:
    """The main game controller for Azul.

    Owns all mutable game state directly — factories, center, bag, discard,
    and the list of players. Enforces move legality, manages round and game
    transitions, and delegates per-player board logic to Player.

    Attributes:
        seed:                 RNG seed used for this game.
        players:              One Player per player, in turn order.
        current_player_index: Index of the player whose turn it is.
        factories:            The factory displays. Empty between rounds.
        center:               Shared tile pool. Holds leftover tiles and the
                              FIRST_PLAYER marker during a round.
        bag:                  Draw pile. Refilled from discard when empty.
        discard:              Tiles removed from completed pattern lines and floors.
        round:                Current round number. 0 before setup_round.
        turn:                 Total turns taken across all rounds.
    """

    def __init__(
        self,
        player_names: list[str] | None = None,
        agents: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        self.seed: int = seed or random.randint(1, 2**31)
        random.seed(self.seed)

        names = player_names or [f"Player {i + 1}" for i in range(PLAYERS)]
        agent_list = agents or ["human"] * PLAYERS

        self.players: list[Player] = [
            Player(name=name, agent=agent) for name, agent in zip(names, agent_list)
        ]
        self.current_player_index: int = 0
        self.factories: list[list[Tile]] = [[] for _ in range(NUMBER_OF_FACTORIES)]
        self.center: list[Tile] = []
        self.bag: list[Tile] = [tile for tile in COLOR_TILES] * TILES_PER_COLOR
        self.discard: list[Tile] = []
        self.round: int = 0
        self.turn: int = 0
        random.shuffle(self.bag)

    @property
    def current_player(self) -> Player:
        """The player whose turn it is."""
        return self.players[self.current_player_index]

    # region Display --------------------------------------------------------

    _PLAYER_COLUMN_GAP = "  "
    _TABLE_LABEL_WIDTH = 3
    _TABLE_CELL_WIDTH = 3

    def __str__(self) -> str:
        """Multi-line display of the full game state.

        Three columns side by side, horizontally aligned row by row:
          - Column 0: tile table (bag counts, colour header, F1-F5, center)
          - Column 1+: one column per player (pattern lines, wall, floor)

        Row alignment:
          row 0  BAG row       name line
          row 1  CLR header    score line
          rows 2-6  F1-F5      pattern rows 1-5
          row 7  CTR           floor line

        The current player's name is prefixed with "> ".

        Example:
            Round 1  |  Bag: 80  Discard: 0
            BAG 18 16 17 16 13  > Player 1 (human)        Player 2 (human)
            CLR  B  Y  R  K  W   0+ 0- 0+ 0= 0             0+ 0- 0+ 0= 0
            F1   .  1  .  1  2      .|.....                    .|.....
            F2   1  1  1  1  .     ..|.....                   ..|.....
            F3   .  2  .  1  1    ...|.....                  ...|.....
            F4   1  .  1  .  2   ....|.....                 ....|.....
            F5   .  .  1  1  2  .....|.....                .....|.....
            CTR  .  .  .  .  .  F  .......|                  .......|
        """
        return "\n".join([self._format_round_line()] + self._format_all_columns())

    def __repr__(self) -> str:
        return str(self)

    def _format_round_line(self) -> str:
        """Return the game header line.

        Format: {round}:{turn:02d} [{seed}]  P1: {name} ({agent}) vs P2: ...

        Example: 1:03 [189204712]  P1: Alice (alphabeta_hard) vs P2: Bob (human)
        """
        player_turn = f"P{self.current_player_index + 1}: {self.current_player.name}"
        return f"R{self.round}:T{self.turn:02d} [{self.seed}] {player_turn}"

    def _tile_table_row(self, label: str, cells: list[str]) -> str:
        """Format one row of the tile table: left-justified label + right-justified
        cells."""
        label_cell = label.ljust(self._TABLE_LABEL_WIDTH)
        count_cells = "".join(c.rjust(self._TABLE_CELL_WIDTH) for c in cells)
        return label_cell + count_cells

    def _count_tiles(self, tile_list: list[Tile]) -> list[str]:
        """Return per-colour count strings in COLOR_TILES order, dot for zero."""
        return [
            str(tile_list.count(color)) if tile_list.count(color) > 0 else "."
            for color in COLOR_TILES
        ]

    def _build_tile_table_lines(self) -> list[str]:
        """Return the 8 tile-table lines: BAG, CLR, F1-F5, CTR.

        Aligns with Player.__str__ rows: BAG/name, CLR/score,
        F1-F5/pattern-rows-1-5, CTR/floor.
        """
        first_player_cell = (
            [TILE_CHAR[Tile.FIRST_PLAYER]] if Tile.FIRST_PLAYER in self.center else []
        )
        return [
            self._tile_table_row("BAG", self._count_tiles(self.bag)),
            self._tile_table_row("CLR", [TILE_CHAR[c] for c in COLOR_TILES]),
            *[
                self._tile_table_row(f"F-{i + 1}", self._count_tiles(factory))
                for i, factory in enumerate(self.factories)
            ],
            self._tile_table_row(
                "CTR", self._count_tiles(self.center) + first_player_cell
            ),
        ]

    def _build_player_lines(self, player_index: int) -> list[str]:
        """Return Player.__str__ lines for one player, prefixing the name
        with "> " when that player is current."""
        lines = str(self.players[player_index]).splitlines()
        if player_index == self.current_player_index:
            lines[0] = f">{lines[0][1:]}"
        return lines

    def _format_all_columns(self) -> list[str]:
        """Zip tile-table and all player columns into aligned rows.

        Each column except the last is padded to its natural maximum line
        width before the gap is appended, so columns stay stable regardless
        of tile counts or score values.
        """
        columns = [self._build_tile_table_lines()] + [
            self._build_player_lines(i) for i in range(len(self.players))
        ]
        row_count = max(len(col) for col in columns)
        column_widths = [max((len(line) for line in col), default=0) for col in columns]
        rows = []
        for row_index in range(row_count):
            parts = []
            for col_index, col in enumerate(columns):
                line = col[row_index] if row_index < len(col) else ""
                is_last_column = col_index == len(columns) - 1
                padded = (
                    line if is_last_column else line.ljust(column_widths[col_index])
                )
                parts.append(padded)
            rows.append(self._PLAYER_COLUMN_GAP.join(parts))
        return rows

    # endregion

    # region Clone ----------------------------------------------------------

    def clone(self) -> "Game":
        """Return a fast independent copy of this game."""
        g = object.__new__(Game)
        g.seed = self.seed
        g.current_player_index = self.current_player_index
        g.round = self.round
        g.turn = self.turn
        g.players = [p.clone() for p in self.players]
        g.factories = [f[:] for f in self.factories]
        g.center = self.center[:]
        g.bag = self.bag[:]
        g.discard = self.discard[:]
        return g

    # endregion

    # region Round setup ----------------------------------------------------

    def _refill_bag(self) -> None:
        """Refill the bag from the discard pile and shuffle it."""
        if not self.discard:
            return
        self.bag.extend(self.discard)
        self.discard.clear()
        random.shuffle(self.bag)
        logger.debug("refilled bag from discard and shuffled")

    def _draw_from_bag(self, count: int = TILES_PER_FACTORY) -> list[Tile]:
        """Draw up to count tiles from the bag, refilling from discard if needed."""
        tiles = []
        for _ in range(count):
            if not self.bag:
                self._refill_bag()
            if not self.bag:
                logger.debug("no tiles remaining in discard or bag")
                return tiles
            tiles.append(self.bag.pop())
        return tiles

    def setup_round(self, factories: list[list[Tile]] | None = None) -> None:
        """Set up factories and center for a new round."""
        self.round += 1
        self.center.append(Tile.FIRST_PLAYER)
        if factories is not None:
            for factory, tiles in zip(self.factories, factories):
                factory.extend(tiles)
        else:
            for factory in self.factories:
                factory.extend(self._draw_from_bag())

    def is_round_over(self) -> bool:
        """Return True when no color tiles remain in any factory or the center."""
        factories_empty = all(len(f) == 0 for f in self.factories)
        center_empty = not any(t in COLOR_TILES for t in self.center)
        return factories_empty and center_empty

    # endregion

    # region Move generation ------------------------------------------------

    def _is_valid_destination(
        self, player: Player, tile: Tile, destination: int
    ) -> bool:
        """Return True if tile can legally be placed at destination for player."""
        if destination == FLOOR:
            return True
        return player.is_tile_valid_for_row(tile, destination)

    def legal_moves(self) -> list[Move]:
        """Return all legal moves for the current player."""
        moves = []
        sources = list(enumerate(self.factories))
        sources.append((CENTER, self.center))
        for source_index, source_tiles in sources:
            tile_options = {t for t in source_tiles if t in COLOR_TILES}
            for tile in tile_options:
                for destination in [*range(BOARD_SIZE), FLOOR]:
                    if self._is_valid_destination(
                        self.current_player, tile, destination
                    ):
                        moves.append(
                            Move(
                                tile=tile,
                                source=source_index,
                                destination=destination,
                            )
                        )
        return moves

    def count_distinct_source_tile_pairs(self) -> int:
        """Return the number of distinct (source, tile) pairs with tiles available."""
        count = 0
        for factory in self.factories:
            count += len({t for t in factory if t != Tile.FIRST_PLAYER})
        count += len({t for t in self.center if t != Tile.FIRST_PLAYER})
        return count

    # endregion

    # region Turn execution -------------------------------------------------

    def _take_from_source(self, move: Move) -> list[Tile]:
        """Remove the chosen tile color and FIRST_PLAYER from the source.

        Sends non-chosen color tiles to the center. Updates move.count and
        move.took_first in place so the move is self-describing after execution.
        """
        source = self.center if move.source == CENTER else self.factories[move.source]
        chosen = [t for t in source if t == move.tile or t == Tile.FIRST_PLAYER]
        leftover = [t for t in source if t in COLOR_TILES and t != move.tile]
        source.clear()
        self.center.extend(leftover)
        move.count = sum(1 for t in chosen if t != Tile.FIRST_PLAYER)
        move.took_first = Tile.FIRST_PLAYER in chosen
        return chosen

    def next_player(self) -> None:
        """Advance to the next player in turn order."""
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def make_move(self, move: Move) -> None:
        """Apply a move for the current player."""
        self.turn += 1
        chosen = self._take_from_source(move)
        self.current_player.place(move.destination, chosen)

    def advance(self, *, skip_setup: bool = False) -> bool:
        """Rotate the current player, then handle any phase transitions.

        Returns True if a round boundary was crossed, False otherwise.
        """
        self.next_player()
        if self.is_round_over():
            self._score_round()
            if self.is_game_over():
                self._score_game()
            elif not skip_setup:
                self.setup_round()
            return True
        return False

    # endregion

    # region End-of-round scoring -------------------------------------------

    def _score_round(self) -> None:
        """Score the end of a round for all players."""
        for index, player in enumerate(self.players):
            if Tile.FIRST_PLAYER in player.floor_line:
                self.current_player_index = index
            self.discard.extend(player.process_round_end())

    # endregion

    # region Game-over detection and end-of-game scoring --------------------

    def is_game_over(self) -> bool:
        """Return True if the game has fully ended — all scoring complete, no more
        rounds."""
        return (
            any(player.has_triggered_game_end() for player in self.players)
            and self.is_round_over()
        )

    def _score_game(self) -> None:
        """Apply end-of-game bonus scoring to all players."""
        for player in self.players:
            player.score += player.bonus

    # endregion


if __name__ == "__main__":
    game = Game(seed=42)
    game.setup_round()
    print(game)
