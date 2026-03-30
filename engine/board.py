# engine/board.py

from engine.tile import Tile
from engine.constants import BOARD_SIZE
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

    pattern_lines: list[list[Tile]] = field(
        default_factory=lambda: [[] for _ in range(BOARD_SIZE)]
    )

    wall: list[list[Tile | None]] = field(
        default_factory=lambda: [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    )

    floor_line: list[Tile] = field(default_factory=list)
