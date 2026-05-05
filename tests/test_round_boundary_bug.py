# tests/test_round_boundary_bug.py
"""Regression test: is_round_boundary must not fire when tiles remain in center."""

from engine.game import Game, Move, CENTER
from engine.constants import Tile, COLUMN_FOR_TILE_IN_ROW
from neural.search_tree import AZNode


def _place(wall, row: int, col: int, tile: Tile) -> None:
    wall[row][col] = tile  # type: ignore[index]


def _make_game_for_bug_report() -> Game:
    game = Game()

    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.bag.clear()
    game.discard.clear()

    game.factories[2].extend([Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.YELLOW])

    game.center.extend(
        [
            Tile.FIRST_PLAYER,
            Tile.BLUE,
            Tile.BLUE,
            Tile.RED,
            Tile.RED,
            Tile.RED,
            Tile.BLACK,
            Tile.BLACK,
        ]
    )

    p1 = game.players[0]
    p1.score = 27

    # Row 0 already scored — WHITE placed on wall, pattern line empty
    p1.pattern_lines[0] = []
    p1.pattern_lines[1] = [Tile.WHITE, Tile.WHITE]
    p1.pattern_lines[2] = [Tile.YELLOW, Tile.YELLOW, Tile.YELLOW]
    p1.pattern_lines[3] = [Tile.BLACK, Tile.BLACK]
    p1.pattern_lines[4] = []

    T = Tile
    # Row 0: all 5 tiles placed (already complete — no pending game end)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLUE][0], T.BLUE)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][0], T.YELLOW)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.RED][0], T.RED)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLACK][0], T.BLACK)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.WHITE][0], T.WHITE)

    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.BLUE][1], T.BLUE)
    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][1], T.YELLOW)
    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.RED][1], T.RED)
    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.BLACK][1], T.BLACK)

    _place(p1.wall, 2, COLUMN_FOR_TILE_IN_ROW[T.BLUE][2], T.BLUE)
    _place(p1.wall, 3, COLUMN_FOR_TILE_IN_ROW[T.WHITE][3], T.WHITE)
    _place(p1.wall, 4, COLUMN_FOR_TILE_IN_ROW[T.RED][4], T.RED)
    _place(p1.wall, 4, COLUMN_FOR_TILE_IN_ROW[T.BLACK][4], T.BLACK)

    p1.floor_line.extend([Tile.FIRST_PLAYER, Tile.YELLOW, Tile.YELLOW])
    p1._update_pending()
    p1._update_penalty()
    p1._update_bonus()

    p2 = game.players[1]
    p2.score = 23
    p2.pattern_lines[0] = []
    p2.pattern_lines[1] = []  # already scored
    p2.pattern_lines[2] = []
    p2.pattern_lines[3] = [Tile.BLUE, Tile.BLUE]
    p2.pattern_lines[4] = []

    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLUE][0], T.BLUE)
    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][0], T.YELLOW)
    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLACK][0], T.BLACK)
    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.WHITE][0], T.WHITE)

    _place(p2.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.WHITE][1], T.WHITE)
    _place(p2.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.BLUE][1], T.BLUE)
    _place(p2.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][1], T.YELLOW)
    _place(p2.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.RED][1], T.RED)
    _place(p2.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.BLACK][1], T.BLACK)

    _place(p2.wall, 2, COLUMN_FOR_TILE_IN_ROW[T.BLACK][2], T.BLACK)
    _place(p2.wall, 2, COLUMN_FOR_TILE_IN_ROW[T.WHITE][2], T.WHITE)
    _place(p2.wall, 2, COLUMN_FOR_TILE_IN_ROW[T.RED][2], T.RED)

    _place(p2.wall, 3, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][3], T.YELLOW)

    _place(p2.wall, 4, COLUMN_FOR_TILE_IN_ROW[T.WHITE][4], T.WHITE)
    _place(p2.wall, 4, COLUMN_FOR_TILE_IN_ROW[T.BLUE][4], T.BLUE)
    p2.floor_line.clear()
    p2._update_pending()
    p2._update_penalty()
    p2._update_bonus()

    game.current_player_index = 0

    return game


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_center_tile_counts_after_setup():
    game = _make_game_for_bug_report()
    blue_count = game.center.count(Tile.BLUE)
    red_count = game.center.count(Tile.RED)
    black_count = game.center.count(Tile.BLACK)
    fp_count = game.center.count(Tile.FIRST_PLAYER)
    assert blue_count == 2, f"expected 2 Blues in center, got {blue_count}"
    assert red_count == 3, f"expected 3 Reds in center, got {red_count}"
    assert black_count == 2, f"expected 2 Blacks in center, got {black_count}"
    assert fp_count == 1, f"expected 1 FIRST_PLAYER in center, got {fp_count}"


def test_is_round_over_false_at_start():
    game = _make_game_for_bug_report()
    assert not game.is_round_over()


def _apply_path(game: Game) -> Game:
    game = game.clone()

    game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=2))
    game.advance()

    game.make_move(Move(source=2, tile=Tile.YELLOW, destination=4))
    game.advance()

    game.make_move(Move(source=CENTER, tile=Tile.RED, destination=0))
    game.advance()

    game.make_move(Move(source=CENTER, tile=Tile.BLACK, destination=3))

    return game


def test_one_blue_remains_after_path():
    game = _make_game_for_bug_report()
    game = _apply_path(game)

    blue_in_center = game.center.count(Tile.BLUE)
    color_tiles_in_center = sum(1 for t in game.center if t != Tile.FIRST_PLAYER)

    assert blue_in_center == 1, (
        f"expected 1 Blue in center after path, got {blue_in_center}. "
        f"Full center: {[t.name for t in game.center]}"
    )
    assert (
        color_tiles_in_center == 1
    ), f"expected exactly 1 color tile in center, got {color_tiles_in_center}"


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
