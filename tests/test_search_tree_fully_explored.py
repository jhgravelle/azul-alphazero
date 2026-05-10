# tests/test_search_tree_fully_explored.py
"""Tests for AZNode._fully_explored and SearchTree early-exit on full exploration.

A node is fully explored when:
  - It is a terminal or round boundary (leaf), OR
  - It is fully expanded AND every child is fully explored.

Once the root is fully explored, is_stable() returns True immediately and
further simulations are skipped.
"""

from engine.constants import Tile
from engine.game import Game
from neural.search_tree import SearchTree, AZNode

# ── Helpers ───────────────────────────────────────────────────────────────────


def _uniform_pv(game, legal):
    n = len(legal)
    return ([1.0 / n] * n if n else []), 0.0


def _build_two_move_position() -> Game:
    """A hand-crafted 2-move end-of-round position.

    Only BLACK and WHITE tiles remain in the center. Each player has four
    nearly-full pattern lines so only row 4 (capacity 5) is available for
    new tiles from the center.
    """
    game = Game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.center.extend([Tile.BLACK, Tile.BLACK, Tile.WHITE, Tile.WHITE])

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
    p2._update_penalty()

    game.current_player_index = 0
    game.round = 1
    return game


def _make_tree(game: Game, simulations: int) -> SearchTree:
    tree = SearchTree(
        policy_value_fn=_uniform_pv,
        simulations=simulations,
        use_heuristic_value=True,
    )
    tree.reset(game)
    return tree


# ── AZNode._fully_explored ────────────────────────────────────────────────────


def test_fresh_node_is_not_fully_explored():
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=0)
    assert tree._root is not None
    assert tree._root._fully_explored is False


def test_round_boundary_node_is_fully_explored():
    """A round boundary node is a leaf — always fully explored."""
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=0)
    root = tree._root
    assert root is not None

    boundary_game = game.clone()
    boundary_game.make_move(boundary_game.legal_moves()[0])
    boundary_game.make_move(boundary_game.legal_moves()[0])
    node = AZNode(game=boundary_game)
    assert node.is_round_boundary is True
    assert node._fully_explored is True


def test_terminal_node_is_fully_explored():
    """A terminal node is always fully explored."""
    game = Game()
    game.setup_round()
    node = AZNode(game=game)
    if node.is_terminal:
        assert node._fully_explored is True


def test_fully_explored_after_sufficient_simulations():
    """The 2-move position has a small tree — should be fully explored quickly."""
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    assert tree._root._fully_explored is True


def test_not_fully_explored_with_zero_simulations():
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=0)
    assert tree._root is not None
    assert tree._root._fully_explored is False


def test_fully_explored_propagates_from_leaves():
    """Children become fully explored before the root does."""
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    for child in tree._root.children:
        assert child._fully_explored is True, f"Child {child.move} not fully explored"


def test_fully_explored_flag_is_stable():
    """Once fully explored, running more simulations keeps it True."""
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    assert tree._root._fully_explored is True
    tree._run_simulations()
    assert tree._root._fully_explored is True


# ── is_stable() early exit ────────────────────────────────────────────────────


def test_is_stable_true_when_fully_explored():
    """is_stable() returns True immediately when root is fully explored."""
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    assert tree._root._fully_explored is True
    assert tree.is_stable() is True


def test_is_stable_does_not_require_batch_count_when_fully_explored():
    """Even with 0 stable batches recorded, is_stable() is True if fully explored."""
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    tree._stable_batches = 0
    assert tree._root is not None
    assert tree._root._fully_explored is True
    assert tree.is_stable() is True


def test_inspector_run_batch_skips_when_fully_explored():
    """_run_simulations is not called when tree is already fully explored."""
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    assert tree._root._fully_explored is True

    visit_count_before = tree._root.visits
    tree._run_simulations()
    visit_count_after = tree._root.visits

    assert visit_count_after == visit_count_before, (
        f"Expected no new visits after full exploration, "
        f"got {visit_count_after - visit_count_before} new visits"
    )


# ── Full game tree ────────────────────────────────────────────────────────────


def test_normal_game_position_not_quickly_fully_explored():
    """A full game position with many moves should not be fully explored in 50 sims."""
    game = Game()
    game.setup_round()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    assert tree._root._fully_explored is False


def test_fully_explored_minimax_matches_brute_force():
    """Once fully explored, minimax value should exactly match hand calculation.

    From our hand analysis:
      Root minimax = +5pts (Black->row5, P2 responds with any White move)
    """
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    assert tree._root._fully_explored is True

    serialized = tree.serialize()
    mm_pts = serialized["minimax_value"] * 50
    assert (
        abs(mm_pts - 5.0) < 0.01
    ), f"Expected exactly +5.0pts minimax, got {mm_pts:.2f}pts"
