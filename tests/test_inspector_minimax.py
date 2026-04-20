# tests/test_inspector_minimax.py
"""End-to-end minimax value tests for the inspector search tree.

Uses a hand-crafted two-move end-of-round position with known ground truth:

    P1 (to move): lines 1-4 full, line 5 empty, floor empty
    P2:           line 1 full, line 2 has 1 black, lines 3-5 partial/empty,
                  floor has FIRST_PLAYER tile
    Center:       2 BLACK, 2 WHITE (all factories empty)

Hand-calculated minimax (P1 perspective, points):
    P1 Black → row 5   0   (P2 best: White → row 5,   net 0)
    P1 Black → floor  -2   (P2 best: White → row 5,   net -2)
    P1 White → row 5   0   (P2 best: Black → row 2 or row 5, net 0)
    P1 White → floor  -2   (P2 best: Black → row 2 or row 5, net -2)

Root minimax = 0pts  (P1 should play Black or White → row 5)
"""

from engine.constants import Tile
from engine.game import Game
from neural.search_tree import AZNode, SearchTree

# ── Position setup ────────────────────────────────────────────────────────────


def _build_position() -> Game:
    """Construct the two-move end-of-round position directly on game state."""
    game = Game()

    # Clear factories and center
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()

    # Center: 2 BLACK, 2 WHITE
    game.state.center.extend(
        [
            Tile.BLACK,
            Tile.BLACK,
            Tile.WHITE,
            Tile.WHITE,
        ]
    )

    # P1: lines 1-4 full, line 5 empty, floor empty
    p1 = game.state.players[0]
    p1.pattern_lines[0] = [Tile.WHITE]  # line 1 full
    p1.pattern_lines[1] = [Tile.BLUE, Tile.BLUE]  # line 2 full
    p1.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]  # line 3 full
    p1.pattern_lines[3] = [
        Tile.YELLOW,
        Tile.YELLOW,
        Tile.YELLOW,
        Tile.YELLOW,
    ]  # line 4 full
    p1.pattern_lines[4] = []  # line 5 empty
    p1.floor_line = []

    # P2: line 1 full, line 2 has 1 black, lines 3-4 partial, line 5 empty
    # floor has FIRST_PLAYER tile
    p2 = game.state.players[1]
    p2.pattern_lines[0] = [Tile.WHITE]  # line 1 full
    p2.pattern_lines[1] = [Tile.BLACK]  # line 2 needs 1 more
    p2.pattern_lines[2] = [Tile.RED, Tile.RED]  # line 3 needs 1 more
    p2.pattern_lines[3] = [Tile.BLUE, Tile.BLUE]  # line 4 needs 2 more
    p2.pattern_lines[4] = []  # line 5 empty
    p2.floor_line = [Tile.FIRST_PLAYER]

    # Walls: all empty (round not yet scored)
    for player in game.state.players:
        player.wall = [[None] * 5 for _ in range(5)]

    # Scores: P1 earned 5, P2 earned 0 (set as base scores)
    # p1.score = 5
    # p2.score = 0

    game.state.current_player = 0
    game.state.round = 1

    return game


# ── Helpers ───────────────────────────────────────────────────────────────────


def _uniform_pv(game, legal):
    n = len(legal)
    return ([1.0 / n] * n if n else []), 0.0


def _make_tree(game: Game, simulations: int = 5000) -> SearchTree:
    tree = SearchTree(
        policy_value_fn=_uniform_pv,
        simulations=simulations,
        use_heuristic_value=True,
    )
    tree.reset(game)
    tree._run_simulations()
    return tree


def _find_child(tree: SearchTree, move_str: str) -> "AZNode | None":
    """Find a root child by its move string."""
    from neural.search_tree import _move_str

    if tree._root is None:
        return None
    for child in tree._root.children:
        if child.move is not None and _move_str(child.move) == move_str:
            return child
    return None


# ── Legal moves ───────────────────────────────────────────────────────────────


def test_position_has_exactly_four_legal_moves():
    """P1 can take 2B or 2W, each to row 5 or floor — 4 moves."""
    game = _build_position()
    legal = game.legal_moves()
    assert len(legal) == 4


def test_position_legal_moves_are_correct_colors():

    game = _build_position()
    legal = game.legal_moves()
    colors = {m.tile for m in legal}
    assert colors == {Tile.BLACK, Tile.WHITE}


def test_position_legal_destinations_are_row5_and_floor():
    from engine.game import FLOOR

    game = _build_position()
    legal = game.legal_moves()
    dests = {m.destination for m in legal}
    assert dests == {4, FLOOR}  # row 5 is index 4


# ── Root minimax ──────────────────────────────────────────────────────────────


def test_root_minimax_is_plus_five():
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    mm_pts = serialized["minimax_value"] * 20
    assert abs(mm_pts - 5.0) < 0.5, f"Expected +5pts, got {mm_pts:.1f}pts"


def test_root_best_move_is_black_to_row5():
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    best_mm = max(c["minimax_value"] for c in serialized["children"])
    best_children = [
        c for c in serialized["children"] if abs(c["minimax_value"] - best_mm) < 0.01
    ]
    best_moves = [c["move"] for c in best_children]
    assert (
        "CTR Black → row 5" in best_moves
    ), f"Expected Black→row5 as best, got: {best_moves}"


# ── Per-branch minimax ────────────────────────────────────────────────────────


def test_black_to_row5_minimax_is_plus_five():
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    child = next(
        (c for c in serialized["children"] if c["move"] == "CTR Black → row 5"), None
    )
    assert child is not None
    mm_pts = child["minimax_value"] * 20
    assert abs(mm_pts - 5.0) < 0.5, f"Expected +5pts, got {mm_pts:.1f}pts"


def test_white_to_row5_minimax_is_plus_four():
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    child = next(
        (c for c in serialized["children"] if c["move"] == "CTR White → row 5"), None
    )
    assert child is not None
    mm_pts = child["minimax_value"] * 20
    assert abs(mm_pts - 4.0) < 0.5, f"Expected +4pts, got {mm_pts:.1f}pts"


def test_black_to_floor_minimax_is_plus_three():
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    child = next(
        (c for c in serialized["children"] if c["move"] == "CTR Black → floor"), None
    )
    assert child is not None
    mm_pts = child["minimax_value"] * 20
    assert abs(mm_pts - 3.0) < 0.5, f"Expected +3pts, got {mm_pts:.1f}pts"


def test_white_to_floor_minimax_is_plus_two():
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    child = next(
        (c for c in serialized["children"] if c["move"] == "CTR White → floor"), None
    )
    assert child is not None
    mm_pts = child["minimax_value"] * 20
    assert abs(mm_pts - 2.0) < 0.5, f"Expected +2pts, got {mm_pts:.1f}pts"


def test_after_black_to_row5_p2_best_is_white_to_row5_or_floor():
    """P2's best responses after Black→row5 are White→row5 or White→floor (tied at
    +5)."""
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    p1_node = next(
        (c for c in serialized["children"] if c["move"] == "CTR Black → row 5"), None
    )
    assert p1_node is not None
    assert len(p1_node["children"]) > 0
    # Both White options give same minimax — either is acceptable
    best_p2 = max(p1_node["children"], key=lambda c: c["visits"])
    assert (
        "White" in best_p2["move"]
    ), f"Expected P2 best to involve White tiles, got: {best_p2['move']}"


def test_after_white_to_row5_p2_best_minimax_is_black_to_row2():
    """After P1 White→row5, P2's minimax-optimal move is Black→row2
    (gives P1 only +4, worse for P1 than Black→floor which gives +5)."""
    game = _build_position()
    tree = _make_tree(game)
    serialized = tree.serialize()
    p1_node = next(
        (c for c in serialized["children"] if c["move"] == "CTR White → row 5"), None
    )
    assert p1_node is not None
    assert len(p1_node["children"]) > 0

    # From P2's perspective (minimising P1), Black→row2 is best for P2
    # because it gives P1 the least gain (+4 vs +5 for other options).
    # The minimax value of p1_node should reflect this: +4pts for P1.
    mm_pts = p1_node["minimax_value"] * 20
    assert abs(mm_pts - 4.0) < 0.5, (
        f"Expected White→row5 minimax to be +4pts (P2 plays Black→row2), "
        f"got {mm_pts:.1f}pts"
    )
