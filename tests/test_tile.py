# tests/test_tile.py

from engine.tile import Tile
from engine.constants import BOARD_SIZE


def test_tile_has_correct_length():
    assert len(Tile) == BOARD_SIZE + 1  # 5 colors + first-player marker


def test_tile_has_correct_number_of_colors():
    color_tiles = [t for t in Tile if t != Tile.FIRST_PLAYER]
    assert len(color_tiles) == BOARD_SIZE


def test_tile_first_player_marker_exists():
    assert Tile.FIRST_PLAYER in Tile
