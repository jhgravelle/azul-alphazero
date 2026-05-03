# engine/constants.py

"""Constants and enums used throughout the Azul engine."""

from enum import Enum, auto
from itertools import accumulate


class Tile(Enum):
    """The five tile colors in Azul, plus the first-player marker.
    Colors are ordered by their column index in the first row of the wall pattern.
    """

    BLUE = auto()
    YELLOW = auto()
    RED = auto()
    BLACK = auto()
    WHITE = auto()
    FIRST_PLAYER = auto()


COLOR_TILES: list[Tile] = [
    t for t in Tile if t != Tile.FIRST_PLAYER
]  # the 5 colors of tiles

# --- Board dimensions and counts ---

BOARD_SIZE = 5  # also equal to the number of tile colors
PLAYERS = 2  # number of players in a new GameState
TILES_PER_COLOR = (
    20  # number of tiles of each color in the bag at the start of the game
)
NUMBER_OF_FACTORIES = PLAYERS * 2 + 1  # number of factories in a new GameState
TILES_PER_FACTORY = 4  # number of tiles placed in each factory at the start of a round
FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]
BONUS_ROW = 2  # points for completing a row on the wall
BONUS_COLUMN = 7  # points for completing a column on the wall
BONUS_TILE = 10  # points for placing all 5 tiles of a color on the wall

FLOOR = -2  # sentinel destination: tiles go to the floor line
CAPACITY: list[int] = [
    row + 1 for row in range(BOARD_SIZE)
]  # max tiles per pattern line row

# --- Wall pattern ---

WALL_PATTERN: list[list[Tile]] = [
    [COLOR_TILES[(col - row) % BOARD_SIZE] for col in range(BOARD_SIZE)]
    for row in range(BOARD_SIZE)
]

# --- Helper constants for scoring ---

# CUMULATIVE_FLOOR_PENALTIES[n] is the total penalty for n tiles on the floor.
# Indexed 0..NUMBER_OF_FACTORIES * TILES_PER_FACTORY so any floor size can be
# looked up directly without capping: CUMULATIVE_FLOOR_PENALTIES[len(floor_line)]
# Penalties beyond len(FLOOR_PENALTIES) repeat the maximum value.
_MAX_FLOOR = NUMBER_OF_FACTORIES * TILES_PER_FACTORY  # theoretical max tiles on floor
_cumulative = list(accumulate(FLOOR_PENALTIES))
_max_penalty = _cumulative[-1]
CUMULATIVE_FLOOR_PENALTIES: list[int] = (
    [0] + _cumulative + [_max_penalty] * (_MAX_FLOOR - len(FLOOR_PENALTIES) + 1)
)

# COLUMN_FOR_TILE_IN_ROW[tile][row] returns the column index for that tile
# on the given wall row. Precomputed from WALL_PATTERN to avoid repeated
# .index() calls in hot scoring paths.
COLUMN_FOR_TILE_IN_ROW: dict[Tile, list[int]] = {
    tile: [WALL_PATTERN[row].index(tile) for row in range(BOARD_SIZE)]
    for tile in COLOR_TILES
}

# --- Cell groups for bonus scoring and completion progress ---

CELLS_BY_ROW: list[list[tuple[int, int]]] = [
    [(row, col) for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)
]
CELLS_BY_COLUMN: list[list[tuple[int, int]]] = [
    [(row, col) for row in range(BOARD_SIZE)] for col in range(BOARD_SIZE)
]
CELLS_BY_TILE: list[list[tuple[int, int]]] = [
    [(row, WALL_PATTERN[row].index(tile)) for row in range(BOARD_SIZE)]
    for tile in COLOR_TILES
]

# --- Display ---

BOARD_SEPARATOR = "|"
TILE_CHAR: dict[Tile | None, str] = {
    None: ".",
    Tile.BLUE: "B",
    Tile.YELLOW: "Y",
    Tile.RED: "R",
    Tile.BLACK: "K",
    Tile.WHITE: "W",
    Tile.FIRST_PLAYER: "F",
}
