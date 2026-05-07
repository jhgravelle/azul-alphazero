from dataclasses import dataclass, field

from engine.constants import (
    BOARD_SEPARATOR,
    BOARD_SIZE,
    BONUS_COLUMN,
    BONUS_ROW,
    BONUS_TILE,
    CAPACITY,
    CELLS_BY_COLUMN,
    CELLS_BY_ROW,
    CELLS_BY_TILE,
    COLUMN_FOR_TILE_IN_ROW,
    CUMULATIVE_FLOOR_PENALTIES,
    FLOOR,
    FLOOR_PENALTIES,
    TILE_CHAR,
    WALL_PATTERN,
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
        pattern_lines: The 5 pattern line rows. Row i holds at most i+1
                       tiles of one color.
        floor_line:    Tiles dropped here incur penalty points at round end.
                       May contain more than the 7 penalty slots — extra
                       tiles are harmless (no additional penalty) but tracked
                       for debugging. See _format_floor for display behavior.
        wall:          The 5x5 placed tile grid (None = empty).

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
    pattern_lines: list[list[Tile]] = field(
        default_factory=lambda: [[] for _ in range(BOARD_SIZE)]
    )
    floor_line: list[Tile] = field(default_factory=list)
    wall: list[list[Tile | None]] = field(
        default_factory=lambda: [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
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
            (CELLS_BY_COLUMN, BONUS_COLUMN),
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
        return self.pattern_lines[row][0] if self.pattern_lines[row] else None

    def _is_line_pending(self, row: int) -> bool:
        """Return True if the pattern line is full and will score this round."""
        return len(self.pattern_lines[row]) == CAPACITY[row]

    def _is_cell_pending(self, row: int, col: int) -> bool:
        """Return True if this wall cell will be placed at the end of the round."""
        return (
            self._is_line_pending(row)
            and self._line_tile(row) == WALL_PATTERN[row][col]
        )

    def _cell_completion(self, row: int, col: int) -> int:
        """Return the fill progress toward this wall cell, in tile units.

        Returns CAPACITY[row] if the cell is already placed, the current
        pattern line length if the line is aimed at this cell, or 0 if
        neither condition holds.
        """
        if self.wall[row][col]:
            return CAPACITY[row]
        if self._line_tile(row) == WALL_PATTERN[row][col]:
            return len(self.pattern_lines[row])
        return 0

    def _is_complete(self, cells: list[tuple[int, int]]) -> bool:
        """Return True if every cell in the group is placed or pending.

        Pending cells count as complete — game end is triggered before
        the round scores, so pending placements are included.
        """
        return all(
            self.wall[row][col] or self._is_cell_pending(row, col) for row, col in cells
        )

    def _pending_cells(self) -> list[tuple[int, int]]:
        """Return all wall cells that will be placed at the end of this round."""
        return [
            (row, col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if self._is_cell_pending(row, col)
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
        while right < BOARD_SIZE and self.wall[row][right]:
            right += 1
        horizontal = right - left - 1
        horizontal = horizontal if horizontal > 1 else 0

        # Vertical: walk up through placed or pending, down through placed only
        above, below = row - 1, row + 1
        while above >= 0 and (
            self.wall[above][col] or self._is_cell_pending(above, col)
        ):
            above -= 1
        while below < BOARD_SIZE and self.wall[below][col]:
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
        col = COLUMN_FOR_TILE_IN_ROW[tile][row]
        if self.wall[row][col] is not None:
            return False
        if len(self.pattern_lines[row]) == CAPACITY[row]:
            return False
        if not self.pattern_lines[row]:
            return True
        return self.pattern_lines[row][0] == tile

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
        for row in range(BOARD_SIZE):
            if not self._is_line_pending(row):
                continue
            tile = self.pattern_lines[row].pop()
            col = COLUMN_FOR_TILE_IN_ROW[tile][row]
            self.wall[row][col] = tile
            discard.extend(self.pattern_lines[row])
            self.pattern_lines[row].clear()
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
            tiles.remove(Tile.FIRST_PLAYER)
        if destination != FLOOR:
            tiles.extend(self.pattern_lines[destination])
            self.pattern_lines[destination] = tiles[: CAPACITY[destination]]
            tiles = tiles[CAPACITY[destination] :]
            if len(self.pattern_lines[destination]) == CAPACITY[destination]:
                self._update_pending()
                self._update_bonus()
        self.floor_line.extend(tiles)
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
        features = [CELLS_BY_ROW, CELLS_BY_COLUMN, CELLS_BY_TILE]
        return [
            [self._completion_progress(cells) for cells in feature]
            for feature in features
        ]

    # endregion

    # region Display --------------------------------------------------------

    def _format_score_line(self) -> str:
        """Format the scoring component line: score+pending-penalty+bonus=earned.

        Each value is padded to 2 characters. penalty is shown negated so
        the displayed number is positive (e.g. -0 unambiguously marks the
        penalty slot even when no tiles are on the floor).
        """
        return (
            f"{self.score:>2d}"
            f" + {self.pending:>2d}"
            f" - {-self.penalty:>2d}"
            f" + {self.bonus:>2d}"
            f" = {self.earned:>2d}"
        )

    def _format_row(self, row: int) -> str:
        """Return a formatted string for one pattern line and its wall row.

        Pattern line is right-aligned to BOARD_SIZE with dots for empty
        slots. Wall row shows placed tiles with dots for empty cells.
        The two halves are joined by BOARD_SEPARATOR.
        """
        empty = TILE_CHAR[None] * (CAPACITY[row] - len(self.pattern_lines[row]))
        filled = TILE_CHAR[self._line_tile(row)] * len(self.pattern_lines[row])
        pattern = "".join([*empty, *filled]).rjust(BOARD_SIZE)
        wall = "".join(TILE_CHAR[self.wall[row][col]] for col in range(BOARD_SIZE))
        chars = BOARD_SEPARATOR.join([pattern, wall])
        return " ".join([c for c in chars])

    def _format_floor(self) -> str:
        """Format the floor line for display.

        Shows up to 7 penalty slots (dots for empty), then BOARD_SEPARATOR
        and any overflow tiles beyond slot 7. Overflow tiles incur no
        additional penalty but are shown for debugging visibility.
        """
        length = len(FLOOR_PENALTIES)
        tile_strs = " ".join(TILE_CHAR[tile] for tile in self.floor_line)
        slots = tile_strs[:length].ljust(length, TILE_CHAR[None])
        overflow = tile_strs[length:]
        chars = BOARD_SEPARATOR.join([slots, overflow])
        return " ".join([c for c in chars])

    def __str__(self) -> str:
        """Multi-line monospaced display of the player board.

        Example output:
            Alice (alphabeta_hard(...))
             5+ 3- 1+ 0= 7
                R|..R..
               .Y|.....
              ...|.....
             WWWW|..W..
            ....K|.....
            FR.....|
        """
        return "\n".join(
            [
                f"{self.name} ({self.agent})".rjust((BOARD_SIZE + 1) * 4),
                self._format_score_line().rjust((BOARD_SIZE + 1) * 4),
                "\n".join(
                    self._format_row(row).rjust((BOARD_SIZE + 1) * 4)
                    for row in range(BOARD_SIZE)
                ),
                self._format_floor().rjust((BOARD_SIZE + 1) * 4),
            ]
        )

    def __repr__(self) -> str:
        return str(self)

    # endregion

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
        p.pattern_lines = [line[:] for line in self.pattern_lines]
        p.floor_line = self.floor_line[:]
        p.wall = [row[:] for row in self.wall]
        return p

    # endregion


if __name__ == "__main__":
    player = Player("Joe")
    player.pattern_lines[3] = [Tile.BLACK] * 2
    print(player)
