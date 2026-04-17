# engine/board.py


from engine.constants import Tile, BOARD_SIZE
from dataclasses import dataclass, field


@dataclass
class Board:
    """Represents a single player's board.

    Attributes:
        score: The player's current score.
        pattern_lines: 5 rows; row i can hold at most i+1 tiles of one color.
        wall: 5x5 grid of placed tiles (None = empty).
        floor_line: Tiles dropped here incur minus-point penalties.
    """

    score: int = 0
    clamped_points: int = 0  # number of points saved by clamping score at 0
    pattern_lines: list[list[Tile]] = field(
        default_factory=lambda: [[] for _ in range(BOARD_SIZE)]
    )
    wall: list[list[Tile | None]] = field(
        default_factory=lambda: [
            [None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
        ]
    )

    floor_line: list[Tile] = field(default_factory=list)

    def clone(self) -> "Board":
        """Return a fast independent copy of this board.

        Uses direct list copies rather than deepcopy to avoid Python's
        object-graph traversal overhead.
        """
        b = object.__new__(Board)
        b.score = self.score
        b.clamped_points = self.clamped_points
        b.wall = [row[:] for row in self.wall]
        b.pattern_lines = [line[:] for line in self.pattern_lines]
        b.floor_line = self.floor_line[:]
        return b
