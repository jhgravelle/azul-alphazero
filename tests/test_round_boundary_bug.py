# tests/test_round_boundary_bug.py
"""Regression test: is_round_boundary must not fire when tiles remain in center.

Scenario from bug report:
  - Only F3 has tiles (B Y R Y); all other factories empty.
  - Center: FIRST_PLAYER + B R K K R R B B  (2B 3R 2K + FP token)
  - After path CTR Blue, F3 Yellow, CTR Red, CTR Black:
    center still holds 1 Blue tile.
  - The node reached after CTR Black must NOT be flagged is_round_boundary.
  - The tree must therefore have a child of that node (CTR Blue → ...).
"""

from engine.game import Game, Move, CENTER
from engine.constants import Tile, COLUMN_FOR_TILE_IN_ROW
from neural.search_tree import AZNode, SearchTree

# ── Helpers ───────────────────────────────────────────────────────────────────


def _place(wall, row: int, col: int, tile: Tile) -> None:
    """Set a wall cell, bypassing the static type checker.

    Board.wall is list[list[Tile | None]] at runtime but pyright narrows
    freshly-constructed [[None]*5 ...] to list[list[None]], causing a false
    positive on direct assignment.  Using this helper keeps assignments
    readable without sprinkling type: ignore across every line.
    """
    wall[row][col] = tile  # type: ignore[index]


def _make_game_for_bug_report() -> Game:
    """Construct the exact game state from the bug report.

    P1 (index 0): score=27, current player at tree root.
    P2 (index 1): score=23.

    Factory layout:
        F3 (index 2): [Blue, Yellow, Red, Yellow]
        All other factories: empty

    Center: [FIRST_PLAYER, Blue, Red, Black, Black, Red, Red, Blue, Blue]
            → 3B 3R 2K + FP token
    """
    game = Game()

    # ── Clear the auto-setup state ────────────────────────────────────────
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.bag.clear()
    game.state.discard.clear()

    # ── Factories ─────────────────────────────────────────────────────────
    # F3 = index 2
    game.state.factories[2].extend([Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.YELLOW])

    # ── Center ────────────────────────────────────────────────────────────
    # Bug report: B R K K R R B B  (2B 3R 2K) + FIRST_PLAYER
    game.state.center.extend(
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

    # ── P1 board (index 0) ────────────────────────────────────────────────
    p1 = game.state.players[0]
    p1.score = 27

    # Pattern lines from display (right-to-left fill, colour is the tile):
    #   row 0 (cap 1): W            → [WHITE]
    #   row 1 (cap 2): W W          → [WHITE, WHITE]
    #   row 2 (cap 3): Y Y Y        → [YELLOW, YELLOW, YELLOW]
    #   row 3 (cap 4): K K . .      → [BLACK, BLACK]   (2 of 4 filled)
    #   row 4 (cap 5): empty
    p1.pattern_lines[0] = [Tile.WHITE]
    p1.pattern_lines[1] = [Tile.WHITE, Tile.WHITE]
    p1.pattern_lines[2] = [Tile.YELLOW, Tile.YELLOW, Tile.YELLOW]
    p1.pattern_lines[3] = [Tile.BLACK, Tile.BLACK]
    p1.pattern_lines[4] = []

    # Wall from display:
    #   row 0: B Y R K .   → cols 0-3 filled
    #   row 1: . B Y R K   → cols 1-4 filled
    #   row 2: . . B . .   → col 2 filled
    #   row 3: . . W . .   → col 2 filled
    #   row 4: . R K . .   → cols 1-2 filled
    T = Tile
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLUE][0], T.BLUE)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][0], T.YELLOW)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.RED][0], T.RED)
    _place(p1.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLACK][0], T.BLACK)

    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.BLUE][1], T.BLUE)
    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][1], T.YELLOW)
    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.RED][1], T.RED)
    _place(p1.wall, 1, COLUMN_FOR_TILE_IN_ROW[T.BLACK][1], T.BLACK)

    _place(p1.wall, 2, COLUMN_FOR_TILE_IN_ROW[T.BLUE][2], T.BLUE)

    _place(p1.wall, 3, COLUMN_FOR_TILE_IN_ROW[T.WHITE][3], T.WHITE)

    _place(p1.wall, 4, COLUMN_FOR_TILE_IN_ROW[T.RED][4], T.RED)
    _place(p1.wall, 4, COLUMN_FOR_TILE_IN_ROW[T.BLACK][4], T.BLACK)

    # Floor: F Y Y  → [FIRST_PLAYER, YELLOW, YELLOW]
    p1.floor_line.extend([Tile.FIRST_PLAYER, Tile.YELLOW, Tile.YELLOW])

    # ── P2 board (index 1) ────────────────────────────────────────────────
    p2 = game.state.players[1]
    p2.score = 23

    # Pattern lines:
    #   row 0: empty
    #   row 1: W W   → [WHITE, WHITE]
    #   row 2: empty
    #   row 3: B B . .  → [BLUE, BLUE]
    #   row 4: empty
    p2.pattern_lines[0] = []
    p2.pattern_lines[1] = [Tile.WHITE, Tile.WHITE]
    p2.pattern_lines[2] = []
    p2.pattern_lines[3] = [Tile.BLUE, Tile.BLUE]
    p2.pattern_lines[4] = []

    # Wall from display:
    #   row 0: B Y . K W
    #   row 1: . B Y R K
    #   row 2: K W . . R
    #   row 3: . . . . Y
    #   row 4: . . . W B
    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLUE][0], T.BLUE)
    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.YELLOW][0], T.YELLOW)
    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.BLACK][0], T.BLACK)
    _place(p2.wall, 0, COLUMN_FOR_TILE_IN_ROW[T.WHITE][0], T.WHITE)

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

    # Floor: empty
    p2.floor_line.clear()

    # ── Current player is P1 (index 0) ───────────────────────────────────
    game.state.current_player = 0

    return game


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_center_tile_counts_after_setup():
    """Sanity check: the constructed game has the expected center contents."""
    game = _make_game_for_bug_report()
    blue_count = game.state.center.count(Tile.BLUE)
    red_count = game.state.center.count(Tile.RED)
    black_count = game.state.center.count(Tile.BLACK)
    fp_count = game.state.center.count(Tile.FIRST_PLAYER)
    assert blue_count == 2, f"expected 2 Blues in center, got {blue_count}"
    assert red_count == 3, f"expected 3 Reds in center, got {red_count}"
    assert black_count == 2, f"expected 2 Blacks in center, got {black_count}"
    assert fp_count == 1, f"expected 1 FIRST_PLAYER in center, got {fp_count}"


def test_is_round_over_false_at_start():
    """Round is not over at the start — F3 and center have tiles."""
    game = _make_game_for_bug_report()
    assert not game.is_round_over()


def _apply_path(game: Game) -> Game:
    """Apply the four moves from the bug report and return the resulting game.

    Move sequence (all cloned — does not mutate the original):
      1. CTR Blue   → Row 3  (P1 takes all Blues from center)
      2. F3  Yellow → Row 5  (P2 takes both Yellows from F3)
      3. CTR Red    → Row 1  (P1 takes all Reds from center, incl. F3 overflow)
      4. CTR Black  → Row 4  (P2 takes all Blacks from center)

    After move 4, center should contain exactly 1 Blue tile.
    """
    game = game.clone()

    # Move 1 — P1: CTR Blue → Row 3 (row index 2)
    game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=2))
    game.advance()  # rotates to P2

    # Move 2 — P2: F3 Yellow → Row 5 (row index 4)
    game.make_move(Move(source=2, tile=Tile.YELLOW, destination=4))
    game.advance()  # rotates to P1

    # Move 3 — P1: CTR Red → Row 1 (row index 0)
    game.make_move(Move(source=CENTER, tile=Tile.RED, destination=0))
    game.advance()  # rotates to P2

    # Move 4 — P2: CTR Black → Row 4 (row index 3)
    game.make_move(Move(source=CENTER, tile=Tile.BLACK, destination=3))
    # Do NOT call advance — we want to inspect the state mid-round.

    return game


def test_one_blue_remains_after_path():
    """After the four moves, exactly one Blue tile should remain in center."""
    game = _make_game_for_bug_report()
    game = _apply_path(game)

    blue_in_center = game.state.center.count(Tile.BLUE)
    color_tiles_in_center = sum(1 for t in game.state.center if t != Tile.FIRST_PLAYER)

    assert blue_in_center == 1, (
        f"expected 1 Blue in center after path, got {blue_in_center}. "
        f"Full center: {[t.name for t in game.state.center]}"
    )
    assert (
        color_tiles_in_center == 1
    ), f"expected exactly 1 color tile in center, got {color_tiles_in_center}"


def test_is_round_over_false_after_path():
    """is_round_over() must return False — 1 Blue tile is still in center."""
    game = _make_game_for_bug_report()
    game = _apply_path(game)
    assert not game.is_round_over(), (
        f"is_round_over() returned True but center still has: "
        f"{[t.name for t in game.state.center]}"
    )


def test_aznode_is_round_boundary_false_after_path():
    """AZNode.is_round_boundary must be False when 1 Blue remains in center."""
    game = _make_game_for_bug_report()
    game = _apply_path(game)

    node = AZNode(game=game)
    assert not node.is_round_boundary, (
        f"AZNode.is_round_boundary was True but center still has: "
        f"{[t.name for t in game.state.center]}"
    )


def test_search_tree_does_not_terminate_after_fourth_move():
    """The search tree must not mark the node after CTR Black as a leaf.

    After the four moves, CTR Blue → (some row) should still be a legal
    move and appear as a child in the tree.
    """
    game = _make_game_for_bug_report()

    def uniform_policy_value(game, legal):
        n = len(legal)
        priors = [1.0 / n] * n if n else []
        return priors, 0.0

    tree = SearchTree(policy_value_fn=uniform_policy_value, simulations=400)
    tree.reset(game)

    # Advance the tree through the first three moves so the root is at
    # the state just before move 4 (P2 to play, CTR Black available).
    move1 = Move(source=CENTER, tile=Tile.BLUE, destination=2)
    move2 = Move(source=2, tile=Tile.YELLOW, destination=4)
    move3 = Move(source=CENTER, tile=Tile.RED, destination=0)
    move4 = Move(source=CENTER, tile=Tile.BLACK, destination=3)

    tree.choose_move(game)
    game.make_move(move1)
    game.advance()
    tree.advance(move1)
    tree.choose_move(game)
    game.make_move(move2)
    game.advance()
    tree.advance(move2)
    tree.choose_move(game)
    game.make_move(move3)
    game.advance()
    tree.advance(move3)

    # Now root is at the state where P2 is about to play CTR Black.
    # Run simulations so the tree explores beyond move 4.
    tree.choose_move(game)

    # Find any CTR Black child — destination depends on P2's board state.
    move4_child = None
    if tree._root is not None:
        for child in tree._root.children:
            if (
                child.move is not None
                and child.move.source == CENTER
                and child.move.tile == Tile.BLACK
            ):
                move4_child = child
                break

    assert move4_child is not None, (
        "CTR Black (any destination) not found as a child of the tree root. "
        "Available children: "
        + str(
            [
                (c.move.source, c.move.tile.name, c.move.destination)
                for c in (tree._root.children if tree._root else [])
            ]
        )
    )

    assert not move4_child.is_round_boundary, (
        "CTR Black node is marked is_round_boundary=True, "
        "but 1 Blue tile still remains in center."
    )

    assert move4_child.children, (
        "CTR Black node has no children — tree treated it as a "
        "terminal/boundary leaf, but CTR Blue → (row) should still be legal."
    )

    # Confirm at least one grandchild move involves taking Blue from center.
    blue_from_center = [
        c
        for c in move4_child.children
        if c.move is not None and c.move.source == CENTER and c.move.tile == Tile.BLUE
    ]
    assert blue_from_center, (
        "No grandchild of CTR Black node takes Blue from center. "
        "Grandchildren: "
        + str(
            [
                (c.move.source, c.move.tile.name, c.move.destination)
                for c in move4_child.children
                if c.move is not None
            ]
        )
    )
