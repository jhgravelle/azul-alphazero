# engine/constants.py
"""Constants and enums used throughout the Azul engine."""

from enum import Enum, auto


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
SIZE = 5  # also equal to the number of tile colors
PLAYERS = 2  # number of players in a new GameState
TILES_PER_COLOR = (
    20  # number of tiles of each color in the bag at the start of the game
)
NUMBER_OF_FACTORIES = PLAYERS * 2 + 1  # number of factories in a new GameState
TILES_PER_FACTORY = 4  # number of tiles placed in each factory at the start of a round
FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]
FLOOR_SIZE = len(FLOOR_PENALTIES)  # number of penalty slots on the floor line
BONUS_ROW = 2  # points for completing a row on the wall
BONUS_COLUMN = 7  # points for completing a column on the wall
BONUS_TILE = 10  # points for placing all 5 tiles of a color on the wall
CENTER = -1  # sentinel source: tiles from the center
FLOOR = -2  # sentinel destination: tiles go to the floor line
CAPACITY: list[int] = [row + 1 for row in range(SIZE)]  # max tiles per pattern line row


# --- Display ---
SPACE = " "
BLANK = ""
EMPTY = "."
SEPARATOR = "|"
MOVE_SOURCE_CENTER = "C"  # center pool source in move strings
MOVE_DEST_FLOOR = "F"  # floor destination in move strings
MOVE_MARKER_NORMAL = "-"  # no first-player tile taken
MOVE_MARKER_FIRST_PLAYER = "+"  # first-player tile also taken
MOVE_MARKER_UNKNOWN = "?"  # move not yet executed (count=0)

# --- Conversions ---
TILE_FOR_CHAR: dict[str, Tile | None] = {
    "B": Tile.BLUE,
    "Y": Tile.YELLOW,
    "R": Tile.RED,
    "K": Tile.BLACK,
    "W": Tile.WHITE,
    "F": Tile.FIRST_PLAYER,
    EMPTY: None,
}
TILE_FOR_ROW_COL: list[list[Tile]] = [
    [COLOR_TILES[(col - row) % SIZE] for col in range(SIZE)] for row in range(SIZE)
]
CHAR_FOR_TILE: dict[Tile | None, str] = {
    tile: char for char, tile in TILE_FOR_CHAR.items()
}
CHAR_FOR_ROW_COL: list[list[str]] = [
    [CHAR_FOR_TILE[TILE_FOR_ROW_COL[row][col]] for col in range(SIZE)]
    for row in range(SIZE)
]
COL_FOR_CHAR_ROW: dict[str, list[int]] = {
    char: [
        next(col for col in range(SIZE) if TILE_FOR_ROW_COL[row][col] == tile)
        for row in range(SIZE)
    ]
    for char, tile in TILE_FOR_CHAR.items()
    if tile is not None and tile != Tile.FIRST_PLAYER
}
COL_FOR_TILE_ROW: dict[Tile, list[int]] = {
    tile: [TILE_FOR_ROW_COL[row].index(tile) for row in range(SIZE)]
    for tile in COLOR_TILES
}

# --- Cell groups for bonus scoring and completion progress ---
CELLS_BY_ROW: list[list[tuple[int, int]]] = [
    [(row, col) for col in range(SIZE)] for row in range(SIZE)
]
CELLS_BY_COL: list[list[tuple[int, int]]] = [
    [(row, col) for row in range(SIZE)] for col in range(SIZE)
]
CELLS_BY_TILE: list[list[tuple[int, int]]] = [
    [(row, TILE_FOR_ROW_COL[row].index(tile)) for row in range(SIZE)]
    for tile in COLOR_TILES
]
