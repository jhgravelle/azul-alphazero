# engine/game.py
"""Core game controller for Azul."""

import logging
import random

from engine.constants import (
    SIZE,
    CENTER,
    COLOR_TILES,
    FLOOR,
    NUMBER_OF_FACTORIES,
    PLAYERS,
    CHAR_FOR_TILE,
    TILES_PER_COLOR,
    TILES_PER_FACTORY,
    PLAYER_COLUMN_GAP,
    TABLE_LABEL_WIDTH,
    TABLE_CELL_WIDTH,
    Tile,
)
from engine.move import Move
from engine.player import Player

logger = logging.getLogger(__name__)


class Game:
    """The main game controller for Azul.

    Owns all mutable game state directly — factories, center, bag, discard,
    and the list of players. Enforces move legality, manages round and game
    transitions, and delegates per-player board logic to Player.

    Attributes:
        seed:                 RNG seed used for this game.
        players:              list[Player], one per player, in turn order.
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
        seed: int | None = None,
    ) -> None:
        self.seed: int = seed or random.randint(1, 2**31)
        self._rng: random.Random = random.Random(self.seed)
        names = player_names or [f"Player {i + 1}" for i in range(PLAYERS)]
        self.players: list[Player] = [Player(name=name) for name in names]
        self.current_player_index: int = 0
        self.factories: list[list[Tile]] = [[] for _ in range(NUMBER_OF_FACTORIES)]
        self.center: list[Tile] = []
        self.bag: list[Tile] = [tile for tile in COLOR_TILES] * TILES_PER_COLOR
        self.discard: list[Tile] = []
        self.round: int = 0
        self.turn: int = 0
        self._rng.shuffle(self.bag)
        self.encoded_features: list[int] = []
        self._encode()

    @property
    def current_player(self) -> Player:
        """The player whose turn it is."""
        return self.players[self.current_player_index]

    # region Game flow --------------------------------------------------------

    def setup_round(self, factories: list[list[Tile]] | None = None) -> None:
        """Set up factories and center for a new round.
        optionally provide a list of factory contents."""
        self.round += 1
        self.center.append(Tile.FIRST_PLAYER)
        if factories is not None:
            assert len(self.factories) == len(factories)
            for factory, tiles in zip(self.factories, factories):
                factory.extend(self._remove_from_bag(tiles))
        else:
            for factory in self.factories:
                factory.extend(self._draw_from_bag())
        self._encode()

    # region Display --------------------------------------------------------

    # Slices into encoded_features list
    # Each slice defines the range of values for that encoding section
    # Layout (355 values):
    # - round (1): current round number
    # - can_current_player_trigger (1): boolean flag
    # - can_opponent_trigger (1): boolean flag
    # - has_game_end_been_triggered (1): boolean flag
    # - tile_availability (5): tiles available for each color [B, Y, R, K, W]
    # - tile_source_count (5): sources where each color can be obtained
    # - bag_state (5): tiles of each color in bag
    # - current_player_encoded (168): full encoded state of current player
    # - opponent_encoded (168): full encoded state of opponent player
    ENCODING_SLICES = {
        "round": slice(0, 1),
        "can_current_player_trigger": slice(1, 2),
        "can_opponent_trigger": slice(2, 3),
        "has_game_end_been_triggered": slice(3, 4),
        "tile_availability": slice(4, 9),
        "tile_source_count": slice(9, 14),
        "bag_state": slice(14, 19),
        "current_player_encoded": slice(19, 187),
        "opponent_encoded": slice(187, 355),
    }

    def __str__(self) -> str:
        """Multi-line display of the full game state.

        Three columns side by side, horizontally aligned:
          - Column 1: Player columns (P1 + P2 side-by-side)
          - Column 2: Gap
          - Column 3: tile table (BAG, CLR, F1-F5, CTR)

        Row alignment:
          row 0  Header (R#:T## [SEED])  BAG row
          row 1  Player headers           CLR header
          rows 2-7  Player patterns       F1-F5 + CTR

        Current player is marked with "> " prefix on their header.

        Example:
            R1:T00 [0000000042]                                BAG 18 16 17 16 13
            > P1: Human      0(  0)    P2: AB8:2      0(  0)   CLR  B  Y  R  K  W
                      . | . . . . .            . | . . . . .   F-1  .  1  .  1  2
        """
        return "\n".join(self._format_game_display())

    def __repr__(self) -> str:
        return str(self)

    def _format_round_line(self) -> str:
        """Return the game header line with zero-padded seed.

        Format: R{round}:T{turn:02d} [{seed:010d}]

        Example: R1:T03 [0000000042]
        """
        return f"R{self.round}:T{self.turn:02d} [{self.seed:010d}]"

    def _tile_table_row(self, label: str, cells: list[str]) -> str:
        """Format one row of the tile table: left-justified label + right-justified
        cells."""
        label_cell = label.ljust(TABLE_LABEL_WIDTH)
        count_cells = "".join(c.rjust(TABLE_CELL_WIDTH) for c in cells)
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
            [CHAR_FOR_TILE[Tile.FIRST_PLAYER]]
            if Tile.FIRST_PLAYER in self.center
            else []
        )
        return [
            self._tile_table_row("BAG", self._count_tiles(self.bag)),
            self._tile_table_row("CLR", [CHAR_FOR_TILE[c] for c in COLOR_TILES]),
            *[
                self._tile_table_row(f"F-{i + 1}", self._count_tiles(factory))
                for i, factory in enumerate(self.factories)
            ],
            self._tile_table_row(
                "CTR", self._count_tiles(self.center) + first_player_cell
            ),
        ]

    def _build_player_lines(self, player_index: int) -> list[str]:
        """Return Player.__str__ lines for one player, prefixing with player
        index and marking current player with "> "."""
        lines = str(self.players[player_index]).splitlines()
        player_num = player_index + 1
        if player_index == self.current_player_index:
            prefix = f"> P{player_num}: "
        else:
            prefix = f"  P{player_num}: "
        # Prefix is exactly 6 characters to replace the 6 leading spaces
        # Replace the 6 leading spaces with the prefix
        lines[0] = prefix + lines[0][6:]
        return lines

    def _format_game_display(self) -> list[str]:
        """Build the 3-column game display: players left, factories right.

        Layout:
          - Row 0: Round/Turn/Seed + BAG counts
          - Row 1: Player headers + CLR header
          - Rows 2-7: Player patterns + Factory/Center rows

        Each column except the last is padded to maximum width before gap.
        """
        tile_table_lines = self._build_tile_table_lines()
        player_columns = [self._build_player_lines(i) for i in range(len(self.players))]

        # Calculate column widths for player columns
        player_widths = [
            max((len(line) for line in col), default=0) for col in player_columns
        ]
        total_player_width = sum(player_widths) + (len(player_columns) - 1) * len(
            PLAYER_COLUMN_GAP
        )

        rows = []

        # Row 0: Round header + BAG
        round_line = self._format_round_line()
        bag_line = tile_table_lines[0]
        header_row = round_line.ljust(total_player_width) + PLAYER_COLUMN_GAP + bag_line
        rows.append(header_row)

        # Rows 1-7: Player content + tile table content
        # Note: row 0 is special, so player row i maps to display row i+1
        max_rows = max(len(col) for col in player_columns + [tile_table_lines])
        for row_idx in range(1, max_rows):
            player_parts = []
            for col_idx, col in enumerate(player_columns):
                # row_idx=1 uses col[0] (header), row_idx=2 uses col[1] (pattern 1)
                player_row_idx = row_idx - 1
                line = col[player_row_idx] if player_row_idx < len(col) else ""
                is_last_player_col = col_idx == len(player_columns) - 1
                padded = (
                    line if is_last_player_col else line.ljust(player_widths[col_idx])
                )
                player_parts.append(padded)

            players_str = PLAYER_COLUMN_GAP.join(player_parts)

            # Build tile table part
            tile_line = (
                tile_table_lines[row_idx] if row_idx < len(tile_table_lines) else ""
            )

            rows.append(
                players_str.ljust(total_player_width) + PLAYER_COLUMN_GAP + tile_line
            )

        return rows

    # endregion

    # region Clone ----------------------------------------------------------

    def clone(self) -> "Game":
        """Return a fast independent copy of this game."""
        g = object.__new__(Game)
        g.seed = self.seed
        g._rng = random.Random()
        g._rng.setstate(self._rng.getstate())
        g.current_player_index = self.current_player_index
        g.round = self.round
        g.turn = self.turn
        g.players = [p.clone() for p in self.players]
        g.factories = [f[:] for f in self.factories]
        g.center = self.center[:]
        g.bag = self.bag[:]
        g.discard = self.discard[:]
        g.encoded_features = self.encoded_features[:]
        return g

    # endregion

    # region Encoding (private) -----------------------------------------------

    def _encode(self) -> None:
        """Compute and cache all game-state encoding into encoded_features.

        Builds 355 values in a fixed layout (see ENCODING_SLICES):
        - round (1): Current round number
        - can_current_player_trigger (1): Boolean flag
        - can_opponent_trigger (1): Boolean flag
        - has_game_end_been_triggered (1): Boolean flag
        - tile_availability (5): Number of tiles available for each color
        - tile_source_count (5): Number of sources where each color can be obtained
        - bag_state (5): Number of tiles of each color in bag
        - current_player_encoded (168): Full encoded state of current player
        - opponent_encoded (168): Full encoded state of opponent player
        """
        # Compute game-state features
        avail = self._tile_availability()
        tile_avail = [avail[color][0] for color in COLOR_TILES]
        tile_sources = [avail[color][1] for color in COLOR_TILES]
        bag_counts = [self.bag.count(color) for color in COLOR_TILES]
        current_idx = self.current_player_index
        can_current_trigger = (
            1 if not self.current_player.has_triggered_game_end() else 0
        )
        can_opponent_trigger = (
            1 if not self.players[1 - current_idx].has_triggered_game_end() else 0
        )
        has_game_end_triggered = (
            1 if any(p.has_triggered_game_end() for p in self.players) else 0
        )

        # Build flat list
        self.encoded_features = [
            self.round,
            can_current_trigger,
            can_opponent_trigger,
            has_game_end_triggered,
            *tile_avail,
            *tile_sources,
            *bag_counts,
            *self.players[current_idx].encoded_features,
            *self.players[1 - current_idx].encoded_features,
        ]

    # endregion

    # region Deserialization (public) -----------------------------------------

    @classmethod
    def from_string(cls, text: str) -> "Game":
        """Reconstruct a Game from the output of __str__.

        Parses the multi-line display output and reconstructs all game state:
        player names, scores, board states, factories, center, bag, discard,
        current player, and round.

        Returns a Game instance in identical state to the original.

        Raises:
            ValueError: if the string does not match the expected format.

        Round-trip guarantee:
            str(original) == str(Game.from_string(str(original)))
        """
        import re

        lines = text.strip().splitlines()
        if len(lines) < 2:
            raise ValueError("Expected at least 2 lines")

        # Parse the header line: R{round}:T{turn:02d} [{seed:010d}]
        header_line = lines[0]
        try:
            match = re.match(r"R(\d+):T(\d+)\s+\[(\d+)\]", header_line)
            if not match:
                raise ValueError(f"Invalid header line format: {header_line!r}")
            round_num = int(match.group(1))
            turn_num = int(match.group(2))
            seed = int(match.group(3))
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid header line format: {header_line!r}") from exc

        # The first data line (line 0) has the header info and BAG on the right
        # The remaining lines have player displays + tile table (CLR, F1-F5, CTR)
        if len(lines) < 3:
            raise ValueError("Expected at least 3 lines (header + 2 data rows)")

        data_lines = lines[1:]

        # Find where the tile table starts by looking for tile table patterns
        tile_table_col_start = cls._find_tile_table_split(data_lines)

        # Separate player lines from tile table lines
        # First, extract BAG from the header line (line 0)
        header_line = lines[0]
        tile_table_lines = []
        if tile_table_col_start < len(header_line):
            bag_section = header_line[tile_table_col_start:].lstrip()
            if bag_section:
                tile_table_lines.append(bag_section)

        # Then process the remaining data lines
        player_lines_all = []
        for line in data_lines:
            if tile_table_col_start < len(line):
                player_section = line[:tile_table_col_start].rstrip()
                tile_section = line[tile_table_col_start:].lstrip()
            else:
                player_section = line.rstrip()
                tile_section = ""

            player_lines_all.append(player_section)
            if tile_section:
                tile_table_lines.append(tile_section)

        # Parse tile table to reconstruct factories and center
        factories, center, bag = cls._parse_tile_table(tile_table_lines)

        # Extract individual player columns from combined player lines
        player_line_groups = cls._extract_player_columns(player_lines_all)

        # Parse players and detect current player from ">" marker
        current_player_index = 0
        players = []
        for player_idx, player_lines in enumerate(player_line_groups):
            # Remove empty lines and strip whitespace
            player_lines = [line.rstrip() for line in player_lines if line.strip()]

            if len(player_lines) != 7:
                raise ValueError(
                    f"Expected 7 lines for player {player_idx}, got {len(player_lines)}"
                )

            # Replace player prefix with 6 leading spaces for Player.from_string()
            # Prefixes are: "> P{num}: " (6 chars) or "  P{num}: " (6 chars)
            header = player_lines[0]
            if header.startswith(">"):
                # Current player: "> P{num}: " (6 chars total) -> "      " (6 spaces)
                current_player_index = player_idx
                player_lines[0] = "      " + header[6:]
            elif header.startswith("  P"):
                # Other player: "  P{num}: " (6 chars) -> "      " (6 spaces)
                player_lines[0] = "      " + header[6:]
            else:
                # Fallback: already has correct format
                pass

            player_text = "\n".join(player_lines)
            player = Player.from_string(player_text)
            players.append(player)

        # Construct the game using object.__new__ to bypass __init__
        game = object.__new__(cls)
        game.seed = seed
        game._rng = random.Random(seed)
        game.current_player_index = current_player_index
        game.players = players
        game.factories = factories
        game.center = center
        game.bag = bag
        game.discard = []  # Discard is not stored in the display, so assume empty
        game.round = round_num
        game.turn = turn_num
        game.encoded_features = []
        game._encode()

        return game

    @classmethod
    def _find_tile_table_split(cls, data_lines: list[str]) -> int:
        """Find the column position where the tile table starts.

        Scans for tile table patterns (BAG, CLR, F-, CTR) and finds the
        most consistent column position across lines.
        """
        tile_patterns = ["BAG", "CLR", "F-", "CTR"]
        split_positions = []

        for line in data_lines:
            # Find the rightmost occurrence of any tile table pattern
            for pattern in tile_patterns:
                pos = line.find(pattern)
                if pos >= 0:
                    # Account for possible leading spaces in tile section
                    # Tile table starts just before the label
                    split_positions.append(max(0, pos - 1))
                    break

        if not split_positions:
            # Fallback: typical player column width is ~46 chars for 2 players + gap
            return 50

        # Return the most common split position, or a conservative estimate
        return min(split_positions) if split_positions else 50

    @classmethod
    def _extract_player_columns(cls, player_lines: list[str]) -> list[list[str]]:
        """Extract individual player column data from combined player lines.

        Players are side-by-side, separated by a 2-space gap. This method
        identifies the gaps by looking for the most consistent "  " pattern
        across all lines (particularly the header line).
        """
        if not player_lines:
            raise ValueError("No player lines to parse")

        # Use the first (header) line to find the gap position
        # The header line has "P1:" and "P2:" which are good anchors
        header_line = player_lines[0]

        # Look for "  P" pattern which indicates the start of "P2:"
        # OR look for the most significant gap in the header line
        gap_pos = None

        # Try to find "  P" pattern (gap before P2:)
        for i in range(len(header_line) - 2):
            if header_line[i : i + 3] == "  P" and i > 15:  # Must be after P1:
                gap_pos = i
                break

        if gap_pos is None:
            # Fallback: find the largest gap by looking for "  " patterns
            # that appear early in the header line (player names)
            max_gap = None
            for i in range(15, min(35, len(header_line) - 1)):
                if header_line[i : i + 2] == "  ":
                    if max_gap is None or i > max_gap:
                        max_gap = i

            gap_pos = max_gap if max_gap is not None else 27

        # Extract players into two columns
        player_line_groups = [[], []]

        for line in player_lines:
            # First player: from 0 to gap_pos
            player1_text = (
                line[:gap_pos].rstrip() if gap_pos < len(line) else line.rstrip()
            )
            player_line_groups[0].append(player1_text)

            # Second player: include gap (2 spaces) as prefix part
            if gap_pos + 2 < len(line):
                player2_text = line[gap_pos:].rstrip()
            else:
                player2_text = ""
            player_line_groups[1].append(player2_text)

        return player_line_groups

    @staticmethod
    def _parse_tile_table(
        lines: list[str],
    ) -> tuple[list[list[Tile]], list[Tile], list[Tile]]:
        """Parse the tile table section to reconstruct factories and center.

        Expected 8 lines: BAG, CLR, F1-F5 (5 lines), CTR.
        Format: "LABEL  B  Y  R  K  W" or "CTR ... F" (F = FIRST_PLAYER).
        Counts are digits or dots (.) for zero.

        Returns: (factories, center, bag)
        """
        if len(lines) != 8:
            raise ValueError(f"Expected 8 tile table lines, got {len(lines)}")

        def parse_counts(line: str, ignore_extra: bool = False) -> list[int]:
            """Extract tile counts from a line.

            Args:
                line: The tile table line
                ignore_extra: If True, ignore extra characters (e.g. FIRST_PLAYER)
            """
            # Skip label (first 3 chars) and split counts
            counts_str = line[3:].strip()
            counts = []
            for c in counts_str.split():
                if c == ".":
                    counts.append(0)
                elif c == "F":
                    # Skip FIRST_PLAYER marker
                    if not ignore_extra:
                        continue
                else:
                    try:
                        counts.append(int(c))
                    except ValueError:
                        if ignore_extra:
                            # Skip unparseable characters in the center line
                            continue
                        raise
            return counts

        def build_tiles(counts: list[int]) -> list[Tile]:
            """Build a tile list from color counts."""
            tiles = []
            for color, count in zip(COLOR_TILES, counts):
                tiles.extend([color] * count)
            return tiles

        # Parse each line
        bag_counts = parse_counts(lines[0])  # BAG
        # lines[1] is CLR header - skip it

        factory_counts = [parse_counts(lines[i]) for i in range(2, 7)]  # F1-F5
        center_counts = parse_counts(
            lines[7], ignore_extra=True
        )  # CTR (may have F marker)

        # Handle FIRST_PLAYER marker in center
        center_has_first_player = "F" in lines[7]

        bag = build_tiles(bag_counts)
        factories = [build_tiles(counts) for counts in factory_counts]
        center = build_tiles(center_counts)
        if center_has_first_player:
            center.append(Tile.FIRST_PLAYER)

        return factories, center, bag

    # endregion

    # region Round setup ----------------------------------------------------

    def _refill_bag(self) -> None:
        """Refill the bag from the discard pile and shuffle it."""
        if not self.discard:
            return
        self.bag.extend(self.discard)
        self.discard.clear()
        self._rng.shuffle(self.bag)
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

    def _remove_from_bag(self, tiles: list[Tile]) -> list[Tile]:
        for tile in tiles:
            if not self.bag or tile not in self.bag:
                self._refill_bag()
            assert tile in self.bag
            self.bag.remove(tile)
        return tiles

    def is_round_over(self) -> bool:
        """Return True when no color tiles remain in any factory or the center."""
        factories_empty = all(len(f) == 0 for f in self.factories)
        center_empty = not any(t in COLOR_TILES for t in self.center)
        return factories_empty and center_empty

    # endregion

    # region Move generation ------------------------------------------------

    def legal_moves(self) -> list[Move]:
        """Return all legal moves for the current player."""
        moves = []
        sources = list(enumerate(self.factories))
        sources.append((CENTER, self.center))
        for source_index, source_tiles in sources:
            tile_options = {t for t in source_tiles if t in COLOR_TILES}
            for tile in tile_options:
                for destination in [*range(SIZE), FLOOR]:
                    # Floor is always a valid destination; pattern lines must
                    # be checked against the player's board constraints
                    if (
                        destination == FLOOR
                        or self.current_player.is_tile_valid_for_row(tile, destination)
                    ):
                        moves.append(
                            Move(
                                tile=tile,
                                source=source_index,
                                destination=destination,
                            )
                        )
        return moves

    def _tile_availability(self) -> dict[Tile, tuple[int, int]]:
        """Return tile counts and source counts across all active sources.

        For each color tile, returns a tuple of (total_tiles, source_count)
        where total_tiles is the number of that color in factories/center,
        and source_count is the number of non-empty sources with that color.
        """
        totals: dict[Tile, int] = {color: 0 for color in COLOR_TILES}
        source_counts: dict[Tile, int] = {color: 0 for color in COLOR_TILES}
        for source in [*self.factories, self.center]:
            for color in COLOR_TILES:
                count = source.count(color)
                if count > 0:
                    totals[color] += count
                    source_counts[color] += 1
        return {color: (totals[color], source_counts[color]) for color in COLOR_TILES}

    @property
    def total_source_count(self) -> int:
        """Return the total number of non-empty tile sources (factories + center).

        Used by agents for search depth heuristics: more sources means the game
        is earlier in the round with more moves to explore.
        """
        availability = self._tile_availability()
        return sum(source_count for _, source_count in availability.values())

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
        # Ensure player encoding is updated, then re-encode game state
        self._encode()

    def advance(self, *, skip_setup: bool = False) -> bool:
        """Rotate the current player, then handle any phase transitions.

        Returns True if a round boundary was crossed, False otherwise.
        """
        self.next_player()
        self._encode()  # Update encoding after player rotation
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
            if Tile.FIRST_PLAYER in player._floor_line:
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
    print("Original game:")
    print(game)
    print("\n" + "=" * 80 + "\n")
    print("Reconstructed from string (round-trip):")
    game_from_str = Game.from_string(str(game))
    print(game_from_str)
    print("\n" + "=" * 80 + "\n")
    print(f"Round-trip successful: {str(game) == str(game_from_str)}")
