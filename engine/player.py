# engine/player.py
from dataclasses import dataclass, field

from engine.constants import (
    BLANK,
    BONUS_COLUMN,
    BONUS_ROW,
    BONUS_TILE,
    CAPACITY,
    CHAR_FOR_TILE,
    COL_FOR_CHAR_ROW,
    COL_FOR_TILE_ROW,
    COLOR_TILES,
    CELLS_BY_COL,
    CELLS_BY_ROW,
    CELLS_BY_TILE,
    EMPTY,
    FLOOR,
    FLOOR_PENALTIES,
    FLOOR_SIZE,
    SEPARATOR,
    SIZE,
    SPACE,
    TILE_FOR_CHAR,
    TILE_FOR_ROW_COL,
    Tile,
)


@dataclass
class Player:
    """A single player in an Azul game.

    Owns all board state directly — pattern lines, wall, floor, and score.
    Carries a display name that survives cloning so recordings and logs
    stay meaningful.

    Scoring components are cached and updated surgically after each move
    so that earned (the projected end-of-round score) is cheap to read.
    Only the acting player's components are recomputed per move.

    Attributes:
        name:           Display name, e.g. "Player 1" or "Alice". Recommended
                        to be 8 chars or less and unique across players/models
                        for best display and debugging, but not enforced.
        score:          Confirmed score from completed rounds (always >= 0).
        pending:        Cached placement points for full pattern lines this
                        round. Updated when a pattern line completes.
        penalty:        Cached floor line penalty (negative or zero). Updated
                        when a tile hits the floor.
        bonus:          Cached end-of-game bonus guaranteed by the current
                        wall state. Updated when a pattern line completes;
                        persists across rounds until replaced by the next
                        _update_bonus call. Not included in _update_score —
                        only applied at game end.
        _pattern_grid:  Grid mirroring the wall pattern. Each cell holds the
                        count of tiles in that pattern line slot. At most one
                        cell per row has a value > 0.
        _floor_line:    Tiles dropped here incur penalty points at round end.
                        May contain more than FLOOR_SIZE penalty slots —
                        extra tiles are harmless (no additional penalty) but
                        tracked for debugging. See _format_floor for display.
        _wall:          The 5x5 placed tile grid (0 = empty, 1 = full).

    Properties:
        earned:         score + pending + penalty + bonus — the projected
                        total if the round ended now. Never stored; always
                        computed from the four cached components above.
    """

    name: str
    score: int = 0
    pending: int = 0
    penalty: int = 0
    bonus: int = 0
    _pattern_grid: list[list[int]] = field(
        default_factory=lambda: [[0] * SIZE for _ in range(SIZE)]
    )
    _floor_line: list[Tile] = field(default_factory=list)
    _wall: list[list[int]] = field(
        default_factory=lambda: [[0] * SIZE for _ in range(SIZE)]
    )

    # region Earned score ---------------------------------------------------

    @property
    def earned(self) -> int:
        """Projected score if the round ended now.

        Combines confirmed score with all cached components. Bonus is
        included prospectively — it reflects end-of-game points already
        guaranteed by the wall, even though they are not applied until
        game end.
        """
        return self.score + self.pending + self.penalty + self.bonus

    # endregion

    # region Cache updates --------------------------------------------------

    def _update_score(self) -> None:
        """Confirm this round's placement and penalty points into score.

        Adds pending and penalty into score (clamped to zero minimum), then
        resets both to zero. Bonus is intentionally not touched here — it
        accumulates across rounds and is only applied at game end.
        """
        self.score = max(0, self.score + self.pending + self.penalty)
        self.pending = 0
        self.penalty = 0

    def _update_pending(self) -> None:
        """Recompute pending from all currently full pattern lines.

        Called whenever a pattern line completes. Sums adjacency scores
        for every cell that will be placed at round end.
        """
        self.pending = sum(
            self._adjacency_count(row, col) for row, col in self._pending_cells()
        )

    def _update_penalty(self) -> None:
        """Recompute the floor line penalty from the current floor length.

        Uses the precomputed cumulative table — a single index lookup.
        Result is always zero or negative.
        """
        self.penalty = sum(FLOOR_PENALTIES[0 : len(self._floor_line)])

    def _update_bonus(self) -> None:
        """Recompute the prospective end-of-game bonus from the current wall.

        Called whenever a pattern line completes (wall may gain a tile at
        round end). Replaces the previous cached value entirely.

        Awards +BONUS_ROW per completed row, +BONUS_COLUMN per completed
        column, +BONUS_TILE per tile color with all 5 placements on the wall.
        Pending cells count as complete for this purpose.
        """
        bonus_features = [
            (CELLS_BY_ROW, BONUS_ROW),
            (CELLS_BY_COL, BONUS_COLUMN),
            (CELLS_BY_TILE, BONUS_TILE),
        ]
        self.bonus = sum(
            bonus
            for feature, bonus in bonus_features
            for cells in feature
            if self._is_complete(cells)
        )

    # endregion

    # region Wall and pattern line inspection -------------------------------

    def _line_tile(self, row: int) -> Tile | None:
        """Return the committed tile color of a pattern line, or None if empty."""
        for col in range(SIZE):
            if self._pattern_grid[row][col] > 0:
                return TILE_FOR_ROW_COL[row][col]
        return None

    def _line_fill(self, row: int) -> int:
        """Return the current fill of a pattern line (0..CAPACITY[row])."""
        return max(self._pattern_grid[row])

    def _cell_completion_units(self, row: int, col: int) -> int:
        """Return the fill progress toward this wall cell, in tile units.

        Returns CAPACITY[row] if the cell is already placed, the current
        pattern line length if the line is aimed at this cell, or 0 if
        neither condition holds.
        """
        return self._wall[row][col] * CAPACITY[row] + self._pattern_grid[row][col]

    def _is_complete(self, cells: list[tuple[int, int]]) -> bool:
        """Return True if every cell in the group is placed or pending.

        Pending cells count as complete — game end is triggered before
        the round scores, so pending placements are included.
        """
        return all(
            self._wall[row][col] or self._pattern_grid[row][col] == CAPACITY[row]
            for row, col in cells
        )

    def _pending_cells(self) -> list[tuple[int, int]]:
        """Return all wall cells that will be placed at the end of this round."""
        return [
            (row, col)
            for row in range(SIZE)
            for col in range(SIZE)
            if self._pattern_grid[row][col] == CAPACITY[row]
        ]

    def _adjacency_count(self, row: int, col: int) -> int:
        """Return the placement score for a tile placed at (row, col).

        Counts contiguous horizontal and vertical runs through the cell,
        including the cell itself. A lone tile with no neighbors scores 1.

        Horizontal: counts placed tiles in the same wall row.
        Vertical: counts placed or pending tiles above (already scored by
        round end when this tile places), but placed tiles only below
        (not yet placed when scoring runs top to bottom in real Azul).
        """
        # Horizontal: walk left and right along placed wall tiles
        left, right = col - 1, col + 1
        while left >= 0 and self._wall[row][left]:
            left -= 1
        while right < SIZE and self._wall[row][right]:
            right += 1
        horizontal = right - left - 1
        horizontal = horizontal if horizontal > 1 else 0

        # Vertical: walk up through placed or pending, down through placed only
        above, below = row - 1, row + 1
        while above >= 0 and (
            self._wall[above][col] or self._pattern_grid[above][col] == CAPACITY[above]
        ):
            above -= 1
        while below < SIZE and self._wall[below][col]:
            below += 1
        vertical = below - above - 1
        vertical = vertical if vertical > 1 else 0

        return horizontal + vertical or 1

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
        if self._wall[row][col] == 1:
            return False
        line_tile = self._line_tile(row)
        if line_tile is None:
            return True
        if line_tile != tile:
            return False
        if self._pattern_grid[row][col] == CAPACITY[row]:
            return False
        return True

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
            if not max(self._pattern_grid[row]) == CAPACITY[row]:
                continue
            tile = self._line_tile(row)
            assert tile
            col = COL_FOR_TILE_ROW[tile][row]
            self._wall[row][col] = 1
            discard.extend([tile] * (CAPACITY[row] - 1))
            self._pattern_grid[row][col] = 0
        self._update_score()
        discard.extend(t for t in self._floor_line if t != Tile.FIRST_PLAYER)
        self._floor_line.clear()
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
        overflow = 0
        overflow_tile = None
        if destination != FLOOR and tiles:
            overflow_tile = tiles[0]
            col = COL_FOR_TILE_ROW[overflow_tile][destination]
            count = len(tiles)
            filled = self._pattern_grid[destination][col]
            overflow = max(0, filled + count - CAPACITY[destination])
            self._pattern_grid[destination][col] = filled + count - overflow
            if self._pattern_grid[destination][col] == CAPACITY[destination]:
                self._update_pending()
                self._update_bonus()
        elif destination == FLOOR:
            self._floor_line.extend(tiles)
        if overflow_tile is not None:
            self._floor_line.extend([overflow_tile] * overflow)
        self._update_penalty()

    def has_triggered_game_end(self) -> bool:
        """Return True if this player has completed at least one wall row.

        Includes pending cells — a full pattern line that will place at
        round end counts as complete. The round still plays out fully
        before game end bonuses are applied.
        """
        return any(self._is_complete(cells) for cells in CELLS_BY_ROW)

    # endregion

    # region Encoding ------------------------------------------------------

    def encode(self) -> list[float]:
        """Return a 150-value flat encoding of this player's state.

        The encoding is from this player's own perspective — it has no
        knowledge of the opponent. The caller assembles per-player encodings
        in (current, opponent) order before adding the game encoding.

        Layout (grouped by section):
            wall                         25
            pattern fill ratios          25
            pattern completion flags      5
            scoring                       5
            first-player token            1
            wall completion progress     15
            top completions               6
            adjacency grid               25
            tiles needed                  1
            incomplete lines count        1
            pattern line demand           5
            wall completion demand       30
            adjacency demand              5
            total used tiles              1
                                        ---
                                        150
        """
        result = [
            *self._encode_wall(),
            *self._encode_pattern_fill_ratios(),
            *self._encode_pattern_completion_flags(),
            *self._encode_scoring(),
            *self._encode_first_player_token(),
            *self._encode_wall_completion_progress(),
            *self._encode_top_completions(),
            *self._encode_adjacency_grid(),
            *self._encode_tiles_needed(),
            *self._encode_incomplete_lines_count(),
            *self._encode_pattern_line_demand(),
            *self._encode_wall_completion_demand(),
            *self._encode_adjacency_demand(),
            *self._encode_total_used_tiles(),
        ]
        assert len(result) == 150, f"Expected 150 encoding values, got {len(result)}"
        return result

    def _encode_wall(self) -> list[float]:
        """25 values — binary wall placement, row-major."""
        return [
            float(self._wall[row][col]) for row in range(SIZE) for col in range(SIZE)
        ]

    def _encode_pattern_fill_ratios(self) -> list[float]:
        """25 values — pattern grid cells / CAPACITY[row], row-major."""
        return [
            self._pattern_grid[row][col] / CAPACITY[row]
            for row in range(SIZE)
            for col in range(SIZE)
        ]

    def _encode_pattern_completion_flags(self) -> list[float]:
        """5 values — 1.0 if the pattern line for that row is full, else 0.0."""
        return [
            1.0 if self._line_fill(row) == CAPACITY[row] else 0.0 for row in range(SIZE)
        ]

    def _encode_scoring(self) -> list[float]:
        """5 values — score, pending, penalty, bonus, earned (each / 100)."""
        return [
            self.score / 100,
            self.pending / 100,
            self.penalty / 100,
            self.bonus / 100,
            self.earned / 100,
        ]

    def _encode_first_player_token(self) -> list[float]:
        """1 value — 1.0 if this player holds the first-player token."""
        return [1.0 if Tile.FIRST_PLAYER in self._floor_line else 0.0]

    def _encode_wall_completion_progress(self) -> list[float]:
        """15 values — completion fraction for each row, column, and tile color."""
        return [
            self._completion_progress(cells)
            for feature in (CELLS_BY_ROW, CELLS_BY_COL, CELLS_BY_TILE)
            for cells in feature
        ]

    def _top_completion_groups(self) -> list[list[tuple[int, int]]]:
        """Return the 6 highest-priority cell groups: 3 rows, 2 cols, 1 color.

        Each feature type is sorted independently by completion fraction,
        descending. The same group ordering is reused by both
        _encode_top_completions and _encode_wall_completion_demand so the
        two encodings refer to the same groups.
        """
        ranked = []
        for feature, take in (
            (CELLS_BY_ROW, 3),
            (CELLS_BY_COL, 2),
            (CELLS_BY_TILE, 1),
        ):
            sorted_groups = sorted(feature, key=self._completion_progress, reverse=True)
            ranked.extend(sorted_groups[:take])
        return ranked

    def _encode_top_completions(self) -> list[float]:
        """6 values — completion fractions of the top 3 rows, top 2 cols, top 1 tile."""
        return [
            self._completion_progress(cells) for cells in self._top_completion_groups()
        ]

    def _encode_adjacency_grid(self) -> list[float]:
        """25 values — _adjacency_count(row, col) / 10 for every wall cell."""
        return [
            self._adjacency_count(row, col) / 10
            for row in range(SIZE)
            for col in range(SIZE)
        ]

    def _started_incomplete_rows(self) -> list[int]:
        """Return rows with a non-empty, non-full pattern line."""
        return [row for row in range(SIZE) if 0 < self._line_fill(row) < CAPACITY[row]]

    def _encode_tiles_needed(self) -> list[float]:
        """1 value — sum of tiles still needed on started incomplete lines / 10."""
        needed = sum(
            CAPACITY[row] - self._line_fill(row)
            for row in self._started_incomplete_rows()
        )
        return [needed / 10]

    def _encode_incomplete_lines_count(self) -> list[float]:
        """1 value — count of pattern lines started but not full (raw int as float)."""
        return [float(len(self._started_incomplete_rows()))]

    def _encode_pattern_line_demand(self) -> list[float]:
        """5 values — for each color, tiles needed to complete lines committed to
        it / 10."""
        demand = {tile: 0 for tile in COLOR_TILES}
        for row in self._started_incomplete_rows():
            line_tile = self._line_tile(row)
            if line_tile is not None:
                demand[line_tile] += CAPACITY[row] - self._line_fill(row)
        return [demand[tile] / 10 for tile in COLOR_TILES]

    def _encode_wall_completion_demand(self) -> list[float]:
        """30 values — for each of 6 top-completion groups, count of empty cells per
        tile / 10.

        A cell is "empty" if neither placed (_wall == 0) nor pending
        (pattern line not yet at capacity for the cell's row/column).
        Six groups × five colors = 30 values.
        """
        result: list[float] = []
        for cells in self._top_completion_groups():
            counts = {tile: 0 for tile in COLOR_TILES}
            for row, col in cells:
                if self._wall[row][col] == 1:
                    continue
                if self._pattern_grid[row][col] == CAPACITY[row]:
                    continue
                counts[TILE_FOR_ROW_COL[row][col]] += 1
            result.extend(counts[tile] / 10 for tile in COLOR_TILES)
        return result

    def _encode_adjacency_demand(self) -> list[float]:
        """5 values — for each color, adjacency sum over its empty wall cells / 10.

        "Empty" = not placed and not pending. A cell whose pattern line is
        already full counts as filled — placing the same color there would
        be illegal (and already happening).
        """
        demand = {tile: 0 for tile in COLOR_TILES}
        for row in range(SIZE):
            for col in range(SIZE):
                if self._wall[row][col] == 1:
                    continue
                if self._pattern_grid[row][col] == CAPACITY[row]:
                    continue
                tile = TILE_FOR_ROW_COL[row][col]
                demand[tile] += self._adjacency_count(row, col)
        return [demand[tile] / 10 for tile in COLOR_TILES]

    def _encode_total_used_tiles(self) -> list[float]:
        """1 value — total tile investment (wall + pattern lines) / 100.

        Each placed wall tile counts as CAPACITY[row] (tiles drawn from the
        bag historically to fill that line). Pattern grid values count as
        themselves. There are 100 tiles in the game.
        """
        wall_tiles = sum(
            self._wall[row][col] * CAPACITY[row]
            for row in range(SIZE)
            for col in range(SIZE)
        )
        pattern_tiles = sum(
            self._pattern_grid[row][col] for row in range(SIZE) for col in range(SIZE)
        )
        return [(wall_tiles + pattern_tiles) / 100]

    def _completion_progress(self, cells: list[tuple[int, int]]) -> float:
        """Return progress toward completing a cell group as a fraction 0..1.

        Numerator is the sum of fill progress for each cell (CAPACITY[row]
        if placed, current line length if aimed at this cell, 0 otherwise).
        Denominator is the sum of CAPACITY[row] for all cells in the group.
        """
        numerator = sum(self._cell_completion_units(row, col) for row, col in cells)
        denominator = sum(CAPACITY[row] for row, _ in cells)
        return numerator / denominator

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
        empty = CHAR_FOR_TILE[None] * (CAPACITY[row] - self._line_fill(row))
        filled = CHAR_FOR_TILE[self._line_tile(row)] * self._line_fill(row)
        pattern = "".join([*empty, *filled]).rjust(6)
        wall = "".join(
            (
                CHAR_FOR_TILE[TILE_FOR_ROW_COL[row][col]]
                if self._wall[row][col]
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

        Parses name and score from the header line, _pattern_grid and _wall
        from the five board rows, and _floor_line from the floor line.
        Calls _update_pending, _update_bonus, and _update_penalty to
        recompute cached scoring components, then asserts that the
        recomputed earned matches the value parsed from the header.

        Raises:
            ValueError: if the string does not match the expected format.
            AssertionError: if recomputed earned does not match parsed earned.
        """
        lines = text.strip().splitlines()
        if len(lines) != 7:
            raise ValueError(f"Expected 7 lines, got {len(lines)}")

        name, score, earned = cls._parse_header(lines[0])
        pattern_grid, wall = cls._parse_board_rows(lines[1:6])
        floor_line = cls._parse_floor(lines[6])

        player = cls(name=name, score=score)
        player._pattern_grid = pattern_grid
        player._wall = wall
        player._floor_line = floor_line
        player._update_pending()
        player._update_bonus()
        player._update_penalty()

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
    def _parse_board_rows(lines: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        """Parse _pattern_grid and _wall from the five board row lines.

        Each line has a spaced pattern side, ' | ', and a spaced wall side.
        Pattern side chars are tile characters or dots. Wall side chars are
        tile characters (placed) or dots (empty) — any character other than
        empty marks the wall as filled. Convenient for hand-entered states.
        """
        pattern_grid = [[0] * SIZE for _ in range(SIZE)]
        wall = [[0] * SIZE for _ in range(SIZE)]
        for row, line in enumerate(lines):
            pattern_str, wall_str = line.replace(SPACE, BLANK).split(SEPARATOR)
            if TILE_FOR_CHAR.get(pattern_str[-1]):
                col = COL_FOR_CHAR_ROW[pattern_str[-1]][row]
                pattern_grid[row][col] = len(pattern_str.replace(EMPTY, BLANK))
            assert (
                len(wall_str) == SIZE
            ), f"Expected wall string of length {SIZE}, got {len(wall_str)}"
            wall[row] = [0 if c == EMPTY else 1 for c in wall_str]
        return pattern_grid, wall

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
        copied with direct list operations to avoid deepcopy overhead.
        Metadata (name) and cached scoring components are copied as plain
        assignments.
        """
        p = object.__new__(Player)
        p.name = self.name
        p.score = self.score
        p.pending = self.pending
        p.penalty = self.penalty
        p.bonus = self.bonus
        p._pattern_grid = [row[:] for row in self._pattern_grid]
        p._floor_line = self._floor_line[:]
        p._wall = [row[:] for row in self._wall]
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
