# engine/player.py
from dataclasses import dataclass, field

from engine.constants import (
    BLANK,
    BONUS_COLUMN,
    BONUS_ROW,
    BONUS_TILE,
    CAPACITY,
    CHAR_FOR_TILE,
    COL_FOR_TILE_ROW,
    EMPTY,
    FLOOR,
    FLOOR_PENALTIES,
    FLOOR_SIZE,
    MAX_USED,
    SEPARATOR,
    SIZE,
    SPACE,
    TILE_FOR_CHAR,
    TILE_FOR_ROW_COL,
    Tile,
)

# Slices into encoded_features list
# Each slice defines the range of values for that encoding section
# Update these if encoding layout changes
ENCODING_SLICES = {
    "wall": slice(0, 25),
    "pending_wall": slice(25, 50),
    "adjacency_grid": slice(50, 75),
    "wall_row_demand": slice(75, 100),
    "wall_col_demand": slice(100, 125),
    "wall_tile_demand": slice(125, 130),
    "pattern_demand": slice(130, 135),
    "pattern_capacity": slice(135, 160),
    "scoring": slice(160, 165),
    "misc": slice(165, 168),
}

# Divisors for normalizing raw int values to [0, 1] range for neural network
ENCODING_DIVISORS = {
    "score": 100,
    "pending": 100,
    "penalty": 100,
    "bonus": 100,
    "earned": 100,
    "wall_completion": 10,
    "adjacency": 10,
    "tiles_needed": 10,
    "demand": 10,
    "bag_count": 20,
    "tile_availability": 20,
    "adjacency_demand": 10,
    "used_tiles": 100,
}


@dataclass
class Player:
    """A single player in an Azul game.

    Owns all board state directly — pattern lines, wall, floor, and score.
    Carries a display name that survives cloning so recordings and logs
    stay meaningful.

    Scoring components (pending, penalty, bonus) are recomputed on demand
    from board state. earned is the sum of all components.

    Properties:
        earned: score + pending + penalty + bonus — the projected total if
            the round ended now. Never stored; always computed.
        pending: Placement points from full pattern lines this round.
        penalty: Floor line penalty (negative or zero).
        bonus: End-of-game bonus guaranteed by current wall state.
        _pattern_grid: Calculated property reconstructing 2D grid from
            _pattern_lines for backward compatibility.
        _wall: Calculated property reconstructing binary grid from _wall_tiles
            for backward compatibility.
    """

    # Display name (recommended 8 chars or less, unique across players/models).
    name: str

    # Confirmed score from completed rounds (always >= 0).
    score: int = 0

    # Pattern lines: list of tile lists, one per row. Each list contains tiles
    # in order placed. Empty if no tiles on that row. Use len() for fill count,
    # append/extend/clear to modify.
    _pattern_lines: list[list[Tile]] = field(
        default_factory=lambda: [[] for _ in range(SIZE)]
    )

    # Floor line: tiles dropped here incur penalty points at round end. May
    # contain more than FLOOR_SIZE slots (extras harmless but tracked).
    _floor_line: list[Tile] = field(default_factory=list)

    # Wall: 5x5 tile placement grid. Each cell holds the placed tile (None
    # if empty).
    _wall_tiles: list[list[Tile | None]] = field(
        default_factory=lambda: [[None for _ in range(SIZE)] for _ in range(SIZE)]
    )

    # Encoded board features: list of 150 raw int values representing all
    # board state metrics for this player. Computed at initialization and
    # after state changes (place, process_round_end). External code accesses
    # this directly; normalization is handled by the encoder/model layer.
    encoded_features: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Compute encoded features on initialization."""
        self._encode()

    @property
    def pending(self) -> int:
        """Placement points from full pattern lines this round.

        Read from encoded_features computed during initialization and after
        state changes.
        """
        return self.encoded_features[ENCODING_SLICES["scoring"].start + 1]

    @property
    def penalty(self) -> int:
        """Floor line penalty (negative or zero).

        Read from encoded_features computed during initialization and after
        state changes.
        """
        return self.encoded_features[ENCODING_SLICES["scoring"].start + 2]

    @property
    def bonus(self) -> int:
        """End-of-game bonus guaranteed by current wall state.

        Read from encoded_features computed during initialization and after
        state changes. Prospective value; applied only at game end.
        """
        return self.encoded_features[ENCODING_SLICES["scoring"].start + 3]

    @property
    def earned(self) -> int:
        """Projected score if the round ended now.

        Combines confirmed score with all computed components. Bonus is
        included prospectively — it reflects end-of-game points already
        guaranteed by the wall, even though they are not applied until
        game end.
        """
        return self.encoded_features[ENCODING_SLICES["scoring"].start + 4]

    # endregion

    # region Pattern grid compatibility property ---------------------------

    # @property
    # def _pattern_grid(self) -> list[list[int]]:
    #     """Calculated property: reconstructs 2D grid from _pattern_lines.

    #     Returns a 5x5 grid where each cell contains the fill count of the
    #     pattern line for that row, at the wall column for that tile color.
    #     Used for backward compatibility during refactoring. New code should
    #     access _pattern_lines directly.
    #     """
    #     grid = [[0] * SIZE for _ in range(SIZE)]
    #     for row, pattern_line in enumerate(self._pattern_lines):
    #         if pattern_line:
    #             tile = pattern_line[-1]
    #             col = COL_FOR_TILE_ROW[tile][row]
    #             grid[row][col] = len(pattern_line)
    #     return grid

    # @property
    # def _wall(self) -> list[list[int]]:
    #     """Calculated property: reconstructs binary grid from _wall_tiles.

    #     Returns a 5x5 grid where each cell contains 1 if a tile is placed,
    #     0 if empty. Used for backward compatibility during refactoring.
    #     New code should access _wall_tiles directly.
    #     """
    #     return [
    #        [1 if self._wall_tiles[row][col] is not None else 0 for col in range(SIZE)]
    #         for row in range(SIZE)
    #     ]

    # endregion

    # region Score updates ---------------------------------------------------

    def _update_score(self) -> None:
        """Confirm this round's placement and penalty points into score.

        Adds pending and penalty into score (clamped to zero minimum).
        Bonus is intentionally not touched here — it accumulates across rounds
        and is only applied at game end via handle_game_end().
        """
        self.score = max(0, self.score + self.pending + self.penalty)

    # endregion

    # region Encoding (private) -----------------------------------------------

    def _encode(self) -> None:
        """Most will first be calcualted by row, col, or tile, count
        cells are by row,col
        demand is by color_idx, then feature index, ie. row 1, col 1, tile 1"""
        wall_cells = [[0] * SIZE for _ in range(SIZE)]  # [row][col] binary
        pending_cells = [[0] * SIZE for _ in range(SIZE)]  # [row][col] binary
        adjacency_cells = [[0] * SIZE for _ in range(SIZE)]  # [row][col] count 0-10
        wall_row_demand = [[0] * SIZE for _ in range(SIZE)]  # [color][row] count
        wall_col_demand = [[0] * SIZE for _ in range(SIZE)]  # [color][col] count
        wall_tile_demand = [0] * SIZE  # [color] count
        pattern_demand = [0] * SIZE  # [color] count
        pattern_capacity = [[0] * SIZE for _ in range(SIZE)]  # [color][row] count

        for row in range(SIZE):
            for col in range(SIZE):
                color_idx = TILE_FOR_ROW_COL[row][col].value
                demand = CAPACITY[row] - self._cell_units(row, col)
                empty = len(self._pattern_lines[row]) == 0
                full = len(self._pattern_lines[row]) == CAPACITY[row]
                committed = demand > 0 and not empty

                wall_cells[row][col] = 1 if self._wall_tiles[row][col] else 0
                pending_cells[row][col] = 1 if committed and full else 0
                adjacency_cells[row][col] = self._adjacency(row, col, for_score=False)
                wall_row_demand[color_idx][row] += demand
                wall_col_demand[color_idx][col] += demand
                wall_tile_demand[color_idx] += demand
                pattern_demand[color_idx] += demand if committed else 0
                pattern_capacity[color_idx][row] += demand if committed or empty else 0

        # Sort columns by total demand, helps model target near complete columns first.
        wall_col_demand.sort(key=lambda x: sum(x))

        # Scoring and Misc calculations
        official_score = self.score
        pending_score = self._pending(pending_cells)
        penalty_score = self._penalty()
        bonus_score = self._bonus(wall_row_demand, wall_col_demand, wall_tile_demand)
        earned_score = official_score + pending_score + penalty_score + bonus_score
        first_player_token = 1 if Tile.FIRST_PLAYER in self._floor_line else 0
        total_used = MAX_USED - sum(wall_tile_demand)
        max_pattern_capacity = self._max_pattern_capacity(pattern_capacity)

        self.encoded_features = self._flatten(
            [
                wall_cells,
                pending_cells,
                adjacency_cells,
                wall_row_demand,
                wall_col_demand,
                wall_tile_demand,
                pattern_demand,
                pattern_capacity,
                official_score,
                pending_score,
                penalty_score,
                bonus_score,
                earned_score,
                first_player_token,
                total_used,
                max_pattern_capacity,
            ]
        )

    def _adjacency(self, row: int, col: int, for_score: bool) -> int:
        """Encode adjacency grid section (25 values).

        For each wall cell, counts contiguous horizontal and vertical runs.
        Lone tiles score 1. Range 0–10.
        Layout: row-major order, one value per wall cell.

        for_score:
            True: this works like scoring at round end counting pending above.
            False: counts adjacency for all pending placements, ie. next rounds state.
        """
        method = self._cell_is_placed if for_score else self._cell_is_full

        # Horizontal: walk left and right
        left, right = col - 1, col + 1
        while left >= 0 and method(row, left):
            left -= 1
        while right < SIZE and method(row, right):
            right += 1
        horizontal = right - left - 1
        horizontal = horizontal if horizontal > 1 else 0

        # Vertical: walk up and down, up always considers pending
        above, below = row - 1, row + 1
        while above >= 0 and self._cell_is_full(above, col):
            above -= 1
        while below < SIZE and method(below, col):
            below += 1
        vertical = below - above - 1
        vertical = vertical if vertical > 1 else 0

        return horizontal + vertical or 1

    def _pending(self, pending_cells):
        return sum(
            self._adjacency(row, col, for_score=True) if pending_cells[row][col] else 0
            for row in range(SIZE)
            for col in range(SIZE)
        )

    def _penalty(self):
        penalty_score = sum(FLOOR_PENALTIES[: len(self._floor_line)])
        return penalty_score

    def _bonus(self, wall_row_demand, wall_col_demand, wall_tile_demand):
        return (
            sum(BONUS_ROW for i in range(SIZE) if sum(wall_row_demand[i]) == 0)
            + sum(BONUS_COLUMN for i in range(SIZE) if sum(wall_col_demand[i]) == 0)
            + sum(BONUS_TILE for i in range(SIZE) if wall_tile_demand[i] == 0)
        )

    def _max_pattern_capacity(self, pattern_capacity):
        max_pattern_capacity = sum(
            max(pattern_capacity[color_idx][row] for color_idx in range(SIZE))
            for row in range(SIZE)
        )

        return max_pattern_capacity  # Flatten all nested lists

    def _flatten(self, data: list) -> list[int]:
        """Flatten a nested list structure of any depth into a single flat list.
        Recursively traverses nested lists and appends all scalar values in order.
        Handles mixed depths — elements can be lists, lists of lists, or scalars.

        Args:
            data: A list containing elements of any depth (nested lists or scalars)

        Returns:
            A flat list of all scalar values in traversal order

        Examples:
            flatten([1, [2, 3], [[4, 5], 6]]) → [1, 2, 3, 4, 5, 6]
            flatten([[1, 2], [3, 4, 5]]) → [1, 2, 3, 4, 5]
            flatten([[[1]]]) → [1]
        """
        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(self._flatten(item))  # Recurse for nested lists
            else:
                result.append(item)  # Append scalar values
        return result

    # endregion

    # def _line_tile(self, row: int) -> Tile | None:
    #     """Return the committed tile color of a pattern line, or None if empty."""
    #     return self._pattern_lines[row][-1] if self._pattern_lines[row] else None

    # def _line_fill(self, row: int) -> int:
    #     """Return the current fill of a pattern line (0..CAPACITY[row])."""
    #     return len(self._pattern_lines[row])

    def _cell_units(self, row: int, col: int) -> int:
        """Return the fill progress toward this wall cell, in tile units.

        Returns CAPACITY[row] if the cell is already placed, the current
        pattern line length if the line is aimed at this cell, or 0 if
        neither condition holds.
        """
        if self._wall_tiles[row][col] is not None:
            return CAPACITY[row]
        elif not self._pattern_lines[row]:
            return 0
        elif self._pattern_lines[row][0] == TILE_FOR_ROW_COL[row][col]:
            return len(self._pattern_lines[row])
        else:
            return 0

    def _cell_is_placed(self, row: int, col: int) -> bool:
        """Return True if the wall cell is already placed."""
        return self._wall_tiles[row][col] is not None

    def _cell_is_full(self, row: int, col: int) -> bool:
        """Return True if the wall cell is placed or will be placed this round."""
        return self._cell_units(row, col) == CAPACITY[row]

    # def _is_complete(self, cells: list[tuple[int, int]]) -> bool:
    #     """Return True if every cell in the group is placed or pending.

    #     Pending cells count as complete — game end is triggered before
    #     the round scores, so pending placements are included.
    #     """
    #     return all(
    #         self._wall_tiles[row][col] is not None
    #         or (
    #             self._pattern_lines[row]
    #             and self._pattern_lines[row][-1] == TILE_FOR_ROW_COL[row][col]
    #             and len(self._pattern_lines[row]) == CAPACITY[row]
    #         )
    #         for row, col in cells
    #     )

    # endregion

    # region Game flow ------------------------------------------------------

    def is_tile_valid_for_row(self, tile: Tile, row: int) -> bool:
        """Return True if tile can legally be placed on the given pattern line row.

        A tile is valid if:
        - The wall cell for that tile in that row is not already filled.
        - The pattern line is not already at full capacity.
        - The pattern line is empty, or already committed to the same color.
        """
        col = COL_FOR_TILE_ROW[tile][row]
        if self._cell_units(row, col) == CAPACITY[row]:  # already placed or full
            return False
        if self._cell_units(row, col) > 0:  # partially full implies correct color
            return True
        if len(self._pattern_lines[row]) == 0:  # we are empty make sure no other tile.
            return True
        return False

    def process_round_end(self) -> list[Tile]:
        """Commit pending pattern lines to the wall and clear the floor.

        For each full pattern line: places the tile on the wall, clears the
        line, and collects the extras (all but one tile) for discard.
        Then calls _update_score() to commit pending and penalty into score.
        Floor tiles are collected for discard (FIRST_PLAYER excluded — it
        returns to the box, not the discard pile).

        Returns:
            All tiles to be added to the game discard pile.
        """
        discard: list[Tile] = []
        for row in range(SIZE):
            if len(self._pattern_lines[row]) != CAPACITY[row]:
                continue
            tile = self._pattern_lines[row][-1]
            col = COL_FOR_TILE_ROW[tile][row]
            self._wall_tiles[row][col] = tile
            discard.extend(self._pattern_lines[row][:-1])
            self._pattern_lines[row].clear()
        self.score = max(0, self.score + self.pending + self.penalty)
        discard.extend(t for t in self._floor_line if t != Tile.FIRST_PLAYER)
        self._floor_line.clear()
        self._encode()
        return discard

    def place(self, destination: int, tiles: list[Tile]) -> None:
        """Place tiles onto the destination pattern line or floor.

        Precondition: tiles contains only valid color tiles for this
        destination — the caller is responsible for checking
        _is_valid_destination before calling. FIRST_PLAYER is handled
        here regardless of destination.

        Overflow beyond line capacity spills to the floor. Cache updates
        (pending, bonus, penalty) are applied after placement.
        """
        if Tile.FIRST_PLAYER in tiles:
            self._floor_line.append(Tile.FIRST_PLAYER)
            tiles = [tile for tile in tiles if tile != Tile.FIRST_PLAYER]
        overflow_tile = None
        if destination != FLOOR and tiles:
            overflow_tile = tiles[0]
            capacity = CAPACITY[destination]
            current_count = len(self._pattern_lines[destination])
            new_count = current_count + len(tiles)
            overflow = max(0, new_count - capacity)
            to_place = new_count - overflow
            self._pattern_lines[destination].extend([overflow_tile] * to_place)
        elif destination == FLOOR:
            self._floor_line.extend(tiles)
        if overflow_tile is not None:
            overflow = max(0, current_count + len(tiles) - CAPACITY[destination])
            self._floor_line.extend([overflow_tile] * overflow)
        self._encode()

    # def has_triggered_game_end(self) -> bool:
    #     """Return True if this player has completed at least one wall row.

    #     Includes pending cells — a full pattern line that will place at
    #     round end counts as complete. The round still plays out fully
    #     before game end bonuses are applied.
    #     """
    #     return any(self._is_complete(cells) for cells in CELLS_BY_ROW)

    # endregion

    # region Display --------------------------------------------------------

    def _format_header(self) -> str:
        """Format the header as a fixed 23-character string.

        Six leading spaces are reserved for Game to overwrite with context
        (e.g. 'P1: ' or '> '). Player name is left-justified to 8 characters,
        truncated with ellipsis if over budget. Score and earned are shown
        as 'score(earned)' right-justified at the end.
        """
        name = self.name if len(self.name) <= 8 else self.name[:7] + "…"
        return f"      {name:<8} {self.score:>3d}({self.earned:>3d})"

    def _format_row(self, row: int) -> str:
        """Return a fixed 23-character string for one pattern line and wall row.

        Pattern side is right-justified to 6 unspaced characters so that
        after spacing every character the separator lands at position 12.
        Wall shows placed tile character for filled cells, dot for empty.
        """
        fill = len(self._pattern_lines[row])
        tile = self._pattern_lines[row][-1] if self._pattern_lines[row] else None
        empty = CHAR_FOR_TILE[None] * (CAPACITY[row] - fill)
        filled = CHAR_FOR_TILE[tile] * fill
        pattern = "".join([*empty, *filled]).rjust(6)
        wall = "".join(
            (
                CHAR_FOR_TILE[self._wall_tiles[row][col]]
                if self._wall_tiles[row][col] is not None
                else CHAR_FOR_TILE[None]
            )
            for col in range(SIZE)
        )
        chars = SEPARATOR.join([pattern, wall])
        return " ".join(c for c in chars)

    def _format_floor(self) -> str:
        """Return a fixed 23-character string for the floor line.

        Shows up to FLOOR_SIZE penalty slots with spaced dots for empty
        tiles, then ' |' and any overflow tiles left-justified in the
        remaining 8 chars.
        """
        floor_chars = [CHAR_FOR_TILE[tile] for tile in self._floor_line]
        slots = floor_chars[:FLOOR_SIZE]
        overflow = floor_chars[FLOOR_SIZE:]
        slots_str = "".join(slots + [CHAR_FOR_TILE[None]] * (FLOOR_SIZE - len(slots)))
        overflow_str = "".join(overflow)
        return f"{slots_str:>11} {SEPARATOR} {overflow_str:<9}"

    def __str__(self) -> str:
        """Multi-line monospaced 23-character-wide display of the player board."""
        return "\n".join(
            [
                self._format_header(),
                "\n".join(self._format_row(row) for row in range(SIZE)),
                self._format_floor(),
            ]
        )

    def __repr__(self) -> str:
        return str(self)

    # endregion

    # region From String ----------------------------------------------------

    @classmethod
    def from_string(cls, text: str) -> "Player":
        """Reconstruct a Player from the output of __str__.

        Parses name and score from the header line, _pattern_lines and _wall
        from the five board rows, and _floor_line from the floor line.
        Recomputes encoded_features and asserts that earned matches parsed value.

        Raises:
            ValueError: if the string does not match the expected format.
            AssertionError: if recomputed earned does not match parsed earned.
        """
        lines = text.strip().splitlines()
        if len(lines) != 7:
            raise ValueError(f"Expected 7 lines, got {len(lines)}")

        name, score, earned = cls._parse_header(lines[0])
        pattern_lines, wall_tiles = cls._parse_board_rows(lines[1:6])
        floor_line = cls._parse_floor(lines[6])

        player = cls(name=name, score=score)
        player._pattern_lines = pattern_lines
        player._wall_tiles = wall_tiles
        player._floor_line = floor_line
        player._encode()

        assert (
            player.earned == earned
        ), f"Recomputed earned {player.earned} does not match parsed earned {earned}"
        return player

    @staticmethod
    def _parse_header(line: str) -> tuple[str, int, int]:
        """Parse name, score, and earned from the header line.

        Score and earned are always at the end in the format ' NNN(NNN)'.
        Name is everything before that suffix, stripped of padding.
        """
        name = line[:-9].strip()
        score = int(line[-8:-5])
        earned = int(line[-4:-1])
        return name, score, earned

    @staticmethod
    def _parse_board_rows(
        lines: list[str],
    ) -> tuple[list[list[Tile]], list[list[Tile | None]]]:
        """Parse _pattern_lines and _wall_tiles from the five board row lines.

        Each line has a spaced pattern side, ' | ', and a spaced wall side.
        Pattern side chars are tile characters or dots. Wall side chars are
        tile characters (placed) or dots (empty) — any character other than
        empty marks the wall as filled. Convenient for hand-entered states.
        """
        pattern_lines: list[list[Tile]] = [[] for _ in range(SIZE)]
        wall_tiles: list[list[Tile | None]] = [
            [None for _ in range(SIZE)] for _ in range(SIZE)
        ]
        for row, line in enumerate(lines):
            pattern_str, wall_str = line.replace(SPACE, BLANK).split(SEPARATOR)
            # Extract tiles from the pattern side (last character indicates which tile)
            pattern_str = pattern_str.replace(EMPTY, BLANK)
            if pattern_str and pattern_str != BLANK:
                last_char = pattern_str[-1]
                if last_char in TILE_FOR_CHAR and TILE_FOR_CHAR[last_char] is not None:
                    tile = TILE_FOR_CHAR[last_char]
                    assert (
                        tile is not None
                    ), f"Invalid tile character '{last_char}' in pattern line"
                    count = len(pattern_str)
                    if count > 0:
                        pattern_lines[row] = [tile] * count
            assert (
                len(wall_str) == SIZE
            ), f"Expected wall string of length {SIZE}, got {len(wall_str)}"
            for col, c in enumerate(wall_str):
                if c != EMPTY:
                    wall_tiles[row][col] = TILE_FOR_ROW_COL[row][col]
        return pattern_lines, wall_tiles

    @staticmethod
    def _parse_floor(line: str) -> list[Tile]:
        """Parse _floor_line from the floor line.

        Expected format: slots (up to FLOOR_SIZE) then ' | ' then overflow
        tiles. Dots are empty slots and are skipped.
        """
        line = line.replace(EMPTY, BLANK)
        line = line.replace(SPACE, BLANK)
        line = line.replace(SEPARATOR, BLANK)
        return [
            tile
            for c in line
            if c in TILE_FOR_CHAR and (tile := TILE_FOR_CHAR[c]) is not None
        ]

    # endregion

    # region Clone ----------------------------------------------------------

    def clone(self) -> "Player":
        """Return a fast independent copy of this player.

        Bypasses __init__ — no default_factory calls. All mutable state is
        copied appropriately. Metadata (name) and cached scoring components
        are copied as plain assignments.
        """
        p = object.__new__(Player)
        p.name = self.name
        p.score = self.score
        p._pattern_lines = [row[:] for row in self._pattern_lines]
        p._floor_line = self._floor_line[:]
        p._wall_tiles = [row[:] for row in self._wall_tiles]
        p.encoded_features = self.encoded_features  # passed by reference - immutable
        # TODO confirm immutability of encoded_features or make a copy if needed
        return p

    # endregion


if __name__ == "__main__":
    player = Player("Alexandria")
    player.place(3, [Tile.BLACK] * 2)
    player.place(FLOOR, [Tile.BLACK] * 9)
    print(player)
    s = str(player)
    player2 = Player.from_string(s)
    print(player2)
