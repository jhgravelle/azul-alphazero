# engine/player.py
from dataclasses import dataclass, field
from typing import Any

from engine.constants import (
    BLANK,
    SEPARATOR,
    COL_FOR_CHAR_ROW,
    EMPTY,
    SIZE,
    BONUS_COLUMN,
    BONUS_ROW,
    BONUS_TILE,
    CAPACITY,
    CELLS_BY_COL,
    CELLS_BY_ROW,
    CELLS_BY_TILE,
    SPACE,
    TILE_FOR_CHAR,
    COL_FOR_TILE_ROW,
    CUMULATIVE_FLOOR_PENALTIES,
    FLOOR,
    CHAR_FOR_TILE,
    TILE_FOR_ROW_COL,
    Tile,
)


@dataclass
class Player:
    """A single player in an Azul game.

    Owns all board state directly — pattern lines, wall, floor, and score.
    Also carries lightweight metadata (name, agent string) that survives
    cloning so recordings and logs stay meaningful.

    Scoring components are cached and updated surgically after each move
    so that earned (the projected end-of-round score) is cheap to read.
    Only the acting player's components are recomputed per move.

    Attributes:
        name:          Display name, e.g. "Player 1" or "Alice".
        agent:         "human", or a string describing the bot and its
                       parameters, e.g.
                       "alphabeta_hard(depths=(3,5,8), thresholds=(20,10))"
                       or "alphazero(checkpoint=gen_0012, sims=1500)".
        score:         Confirmed score from completed rounds (always >= 0).
        pending:       Cached placement points for full pattern lines this
                       round. Updated when a pattern line completes.
        penalty:       Cached floor line penalty (negative or zero). Updated
                       when a tile hits the floor.
        bonus:         Cached end-of-game bonus guaranteed by the current
                       wall state. Updated when a pattern line completes;
                       persists across rounds until replaced by the next
                       _update_bonus call. Not included in update_score —
                       only applied at game end.
        pattern_grid:  Rather than stor the type of tile and a list of how many
                       tiles, we will store a grid that mimics the wall pattern,
                       each cell will have the count of how many tiles are in
                       the pattern line.  there can only be one cell per row
                       with a value > 0
        floor_line:    Tiles dropped here incur penalty points at round end.
                       May contain more than the 7 penalty slots — extra
                       tiles are harmless (no additional penalty) but tracked
                       for debugging. See _format_floor for display behavior.
        wall:          The 5x5 placed tile grid (0 = empty, 1= full).

    Properties:
        earned:        score + pending + penalty + bonus — the projected
                       total if the round ended now. Never stored; always
                       computed from the four cached components above.
    """

    name: str
    agent: str = "human"
    score: int = 0
    pending: int = 0
    penalty: int = 0
    bonus: int = 0
    pattern_grid: list[list[int]] = field(
        default_factory=lambda: [[0] * SIZE for _ in range(SIZE)]
    )
    floor_line: list[Tile] = field(default_factory=list)
    wall: list[list[int]] = field(
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

    def update_score(self) -> None:
        """Confirm this round's placement and penalty points into score.

        Adds pending and penalty into score (clamped to zero minimum), then
        resets both to zero. Bonus is intentionally not touched here — it
        accumulates across rounds and is only applied at game end.
        """
        self.score = max(0, self.score + self.pending + self.penalty)
        self.pending = 0
        self.penalty = 0

    # endregion

    # region Cache updates --------------------------------------------------

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
        self.penalty = CUMULATIVE_FLOOR_PENALTIES[len(self.floor_line)]

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
        """Return the committed color of a pattern line, or None if empty."""
        for col in range(SIZE):
            if self.pattern_grid[row][col] > 0:
                return TILE_FOR_ROW_COL[row][col]
        return None

    def _cell_completion(self, row: int, col: int) -> int:
        """Return the fill progress toward this wall cell, in tile units.

        Returns CAPACITY[row] if the cell is already placed, the current
        pattern line length if the line is aimed at this cell, or 0 if
        neither condition holds.
        """
        return self.wall[row][col] * CAPACITY[row] + self.pattern_grid[row][col]

    def _is_complete(self, cells: list[tuple[int, int]]) -> bool:
        """Return True if every cell in the group is placed or pending.

        Pending cells count as complete — game end is triggered before
        the round scores, so pending placements are included.
        """
        return all(
            self.wall[row][col] or self.pattern_grid[row][col] == CAPACITY[row]
            for row, col in cells
        )

    def _pending_cells(self) -> list[tuple[int, int]]:
        """Return all wall cells that will be placed at the end of this round."""
        return [
            (row, col)
            for row in range(SIZE)
            for col in range(SIZE)
            if self.pattern_grid[row][col] == CAPACITY[row]
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
        while left >= 0 and self.wall[row][left]:
            left -= 1
        while right < SIZE and self.wall[row][right]:
            right += 1
        horizontal = right - left - 1
        horizontal = horizontal if horizontal > 1 else 0

        # Vertical: walk up through placed or pending, down through placed only
        above, below = row - 1, row + 1
        while above >= 0 and (
            self.wall[above][col] or self.pattern_grid[above][col] == CAPACITY[row]
        ):
            above -= 1
        while below < SIZE and self.wall[below][col]:
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
        if self.wall[row][col] == 1:
            return False
        line_tile = self._line_tile(row)
        if line_tile is None:
            return True
        if line_tile != tile:
            return False
        if self.pattern_grid[row][col] == CAPACITY[row]:
            return False
        return True

    def process_round_end(self) -> list[Tile]:
        """Commit pending pattern lines to the wall and clear the floor.

        For each full pattern line: places the tile on the wall, clears the
        line, and collects the extras (all but one tile) for discard.
        Then calls update_score() to commit pending and penalty into score.
        Floor tiles are collected for discard (FIRST_PLAYER excluded — it
        returns to the box, not the discard pile).

        Returns:
            All tiles to be added to the game discard pile.
        """
        discard: list[Tile] = []
        for row in range(SIZE):
            if not max(self.pattern_grid[row]) == CAPACITY[row]:
                continue
            tile = self._line_tile(row)
            assert tile
            col = COL_FOR_TILE_ROW[tile][row]
            self.wall[row][col] = True
            discard.extend([tile] * (CAPACITY[row] - 1))
            self.pattern_grid[row][col] = 0
        self.update_score()
        discard.extend(t for t in self.floor_line if t != Tile.FIRST_PLAYER)
        self.floor_line.clear()
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
            self.floor_line.append(Tile.FIRST_PLAYER)
            tiles = [tile for tile in tiles if tile != Tile.FIRST_PLAYER]
        overflow = 0
        overflow_tile = None
        if destination != FLOOR and tiles:
            overflow_tile = tiles[0]
            col = COL_FOR_TILE_ROW[overflow_tile][destination]
            count = len(tiles)
            filled = self.pattern_grid[destination][col]
            overflow = max(0, filled + count - CAPACITY[destination])
            self.pattern_grid[destination][col] = filled + count - overflow
            if self.pattern_grid[destination][col] == CAPACITY[destination]:
                self._update_pending()
                self._update_bonus()
        elif destination == FLOOR:
            self.floor_line.extend(tiles)
        if overflow_tile is not None:
            self.floor_line.extend([overflow_tile] * overflow)
        self._update_penalty()

    def has_triggered_game_end(self) -> bool:
        """Return True if this player has completed at least one wall row.

        Includes pending cells — a full pattern line that will place at
        round end counts as complete. The round still plays out fully
        before game end bonuses are applied.
        """
        return any(self._is_complete(cells) for cells in CELLS_BY_ROW)

    # endregion

    # region Completion progress --------------------------------------------

    def extract_features(self) -> dict[str, Any]:
        """Return all player-specific cached features."""
        return {
            # Scoring (already cached)
            "score": self.score,
            "pending": self.pending,
            "penalty": self.penalty,
            "bonus": self.bonus,
            "earned": self.earned,  # Property
            # Board state (cheap helpers)
            "wall": [row[:] for row in self.wall],
            "pattern_grid": [row[:] for row in self.pattern_grid],
            "floor_length": len(self.floor_line),
            # Already cached
            "completion_progress": self.encode_completion_progress(),
        }

    def _completion_progress(self, cells: list[tuple[int, int]]) -> float:
        """Return progress toward completing a cell group as a fraction 0..1.

        Numerator is the sum of fill progress for each cell (CAPACITY[row]
        if placed, current line length if aimed at this cell, 0 otherwise).
        Denominator is the sum of CAPACITY[row] for all cells in the group.
        """
        numerator = sum(self._cell_completion(row, col) for row, col in cells)
        denominator = sum(CAPACITY[row] for row, _ in cells)
        return numerator / denominator

    def encode_completion_progress(self) -> list[list[float]]:
        """Return completion progress for all rows, columns, and tile groups.

        Returns a list of three lists: rows, columns, and tile colors.
        Each value is a fraction in 0..1. Used as neural network features.
        """
        features = [CELLS_BY_ROW, CELLS_BY_COL, CELLS_BY_TILE]
        return [
            [self._completion_progress(cells) for cells in feature]
            for feature in features
        ]

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

    def _format_score_line(self) -> str:
        """Format the scoring component line as a fixed 23-character string.

        Each value is right-justified to 3 characters. Operators have a
        leading space only. Penalty is shown negated so the displayed
        number is always positive.
        """
        return (
            f"{self.score:>3d}"
            f" +{self.pending:>3d}"
            f" -{-self.penalty:>3d}"
            f" +{self.bonus:>3d}"
            f" ={self.earned:>3d}"
        )

    def _format_row(self, row: int) -> str:
        """Return a fixed 23-character string for one pattern line and wall row.

        Pattern side is right-justified to 6 unspaced characters so that
        after spacing every character the separator lands at position 12.
        Wall shows placed tile character for filled cells, dot for empty.
        """
        empty = CHAR_FOR_TILE[None] * (CAPACITY[row] - max(self.pattern_grid[row]))
        filled = CHAR_FOR_TILE[self._line_tile(row)] * max(self.pattern_grid[row])
        pattern = "".join([*empty, *filled]).rjust(6)
        wall = "".join(
            (
                CHAR_FOR_TILE[TILE_FOR_ROW_COL[row][col]]
                if self.wall[row][col]
                else CHAR_FOR_TILE[None]
            )
            for col in range(SIZE)
        )
        chars = SEPARATOR.join([pattern, wall])
        return " ".join(c for c in chars)

    def _format_floor(self) -> str:
        """Return a fixed 23-character string for the floor line.

        Shows up to 7 penalty slots with spaced dots for empty tiles, then
        ' |' and any overflow tiles left-justified in the remaining 8 chars.
        """
        penalty_slots = 7
        floor_chars = [CHAR_FOR_TILE[tile] for tile in self.floor_line]
        slots = floor_chars[:penalty_slots]
        overflow = floor_chars[penalty_slots:]
        slots_str = "".join(
            slots + [CHAR_FOR_TILE[None]] * (penalty_slots - len(slots))
        )
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

    # region From String
    @classmethod
    def from_string(cls, text: str) -> "Player":
        """Reconstruct a Player from the output of __str__.

        Parses name and score from the header line, pattern_grid and wall
        from the five board rows, and floor_line from the floor line.
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
        player.pattern_grid = pattern_grid
        player.wall = wall
        player.floor_line = floor_line
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
        """Parse pattern_grid and wall from the five board row lines.

        Each line has a spaced pattern side, ' | ', and a spaced wall side.
        Pattern side chars are tile characters or dots. Wall side chars are
        tile characters (placed) or dots (empty)., any character other than
        empty marks the wall as filled.  Convienient for hand entered states.
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
        """Parse floor_line from the floor line.

        Expected format: slots (up to 7) then ' | ' then overflow tiles.
        Dots are empty slots and are skipped.
        """
        line = line.replace(EMPTY, BLANK)
        line = line.replace(SPACE, BLANK)
        line = line.replace(SEPARATOR, BLANK)
        return [
            tile
            for c in line
            if c in TILE_FOR_CHAR and (tile := TILE_FOR_CHAR[c]) is not None
        ]

    # end region

    # region Clone ----------------------------------------------------------

    def clone(self) -> "Player":
        """Return a fast independent copy of this player.

        Bypasses __init__ — no default_factory calls. All mutable state is
        copied with direct list operations to avoid deepcopy overhead.
        Metadata (name, agent) and cached scoring components are copied
        as plain assignments.
        """
        p = object.__new__(Player)
        p.name = self.name
        p.agent = self.agent
        p.score = self.score
        p.pending = self.pending
        p.penalty = self.penalty
        p.bonus = self.bonus
        p.pattern_grid = [row[:] for row in self.pattern_grid]
        p.floor_line = self.floor_line[:]
        p.wall = [row[:] for row in self.wall]
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
