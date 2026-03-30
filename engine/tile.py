# engine/tile.py

from enum import Enum, auto


class Tile(Enum):
    """The five tile colors in Azul, plus the first-player marker."""

    BLUE = auto()
    YELLOW = auto()
    RED = auto()
    BLACK = auto()
    WHITE = auto()
    FIRST_PLAYER = auto()


COLORS = [t for t in Tile if t != Tile.FIRST_PLAYER]  # the 5 colors of tiles
