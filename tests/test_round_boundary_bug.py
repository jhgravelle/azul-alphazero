# tests/test_round_boundary_bug.py
"""Regression test: is_round_boundary must not fire when tiles remain in center."""

from engine.game import Game, Move, CENTER
from engine.constants import Tile, COL_FOR_TILE_ROW
from neural.search_tree import AZNode


def _place_wall(wall: list[list], row: int, tile: Tile) -> None:
    """Place a tile on the wall by its color and row."""
    col = COL_FOR_TILE_ROW[tile][row]
    wall[row][col] = tile


def _make_game_for_bug_report() -> Game:
    """Create a two-move end-of-round position with tiles still in center.

    This position tests that is_round_boundary does not fire prematurely
    when there are still valid moves available even with few sources.
    """
    game = Game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    game.center.extend([Tile.BLUE, Tile.BLUE, Tile.RED, Tile.BLACK])

    p1 = game.players[0]
    p1.place(0, [Tile.WHITE])
    p1.place(1, [Tile.BLUE, Tile.BLUE])
    p1.place(2, [Tile.RED, Tile.RED, Tile.RED])
    p1.place(3, [Tile.YELLOW, Tile.YELLOW, Tile.YELLOW, Tile.YELLOW])

    p2 = game.players[1]
    p2.place(0, [Tile.WHITE])
    p2.place(1, [Tile.BLACK])
    p2.place(2, [Tile.RED, Tile.RED])
    p2.place(3, [Tile.BLUE, Tile.BLUE])
    p2._floor_line.append(Tile.FIRST_PLAYER)
    p2._encode()

    game.current_player_index = 0
    return game


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_center_tile_counts_after_setup():
    game = _make_game_for_bug_report()
    blue_count = game.center.count(Tile.BLUE)
    red_count = game.center.count(Tile.RED)
    black_count = game.center.count(Tile.BLACK)
    assert blue_count == 2, f"expected 2 Blues in center, got {blue_count}"
    assert red_count == 1, f"expected 1 Red in center, got {red_count}"
    assert black_count == 1, f"expected 1 Black in center, got {black_count}"


def test_is_round_over_false_at_start():
    game = _make_game_for_bug_report()
    assert not game.is_round_over()


def _apply_path(game: Game) -> Game:
    game = game.clone()

    game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=4))
    game.advance()

    game.make_move(Move(source=CENTER, tile=Tile.RED, destination=4))

    return game


def test_tiles_remain_after_path():
    game = _make_game_for_bug_report()
    game = _apply_path(game)

    color_tiles_in_center = sum(1 for t in game.center if t != Tile.FIRST_PLAYER)

    assert color_tiles_in_center > 0, (
        f"expected tiles to remain in center after path, "
        f"got {color_tiles_in_center}. center: {[t.name for t in game.center]}"
    )


def test_is_round_over_false_after_path():
    game = _make_game_for_bug_report()
    game = _apply_path(game)
    assert not game.is_round_over(), (
        f"is_round_over() returned True but center still has: "
        f"{[t.name for t in game.center]}"
    )


def test_aznode_is_round_boundary_false_after_path():
    game = _make_game_for_bug_report()
    game = _apply_path(game)

    node = AZNode(game=game)
    assert not node.is_round_boundary, (
        f"AZNode.is_round_boundary was True but center still has: "
        f"{[t.name for t in game.center]}"
    )
