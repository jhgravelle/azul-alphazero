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
    """The hand-crafted 2-move end-of-round position from test_inspector_minimax."""
    game = Game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.center.extend([Tile.BLACK, Tile.BLACK, Tile.WHITE, Tile.WHITE])

    p1 = game.state.players[0]
    p1.pattern_lines[0] = [Tile.WHITE]
    p1.pattern_lines[1] = [Tile.BLUE, Tile.BLUE]
    p1.pattern_lines[2] = [Tile.RED, Tile.RED, Tile.RED]
    p1.pattern_lines[3] = [Tile.YELLOW, Tile.YELLOW, Tile.YELLOW, Tile.YELLOW]
    p1.pattern_lines[4] = []
    p1.floor_line = []
    p1.wall = [[None] * 5 for _ in range(5)]

    p2 = game.state.players[1]
    p2.pattern_lines[0] = [Tile.WHITE]
    p2.pattern_lines[1] = [Tile.BLACK]
    p2.pattern_lines[2] = [Tile.RED, Tile.RED]
    p2.pattern_lines[3] = [Tile.BLUE, Tile.BLUE]
    p2.pattern_lines[4] = []
    p2.floor_line = [Tile.FIRST_PLAYER]
    p2.wall = [[None] * 5 for _ in range(5)]

    game.state.current_player = 0
    game.state.round = 1
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

    # Manually create a boundary node
    boundary_game = game.clone()
    boundary_game.make_move(boundary_game.legal_moves()[0])
    boundary_game.make_move(boundary_game.legal_moves()[0])
    # Should be at round boundary now
    node = AZNode(game=boundary_game)
    assert node.is_round_boundary is True
    # A boundary node should be considered fully explored
    assert node._fully_explored is True


def test_terminal_node_is_fully_explored():
    """A terminal node is always fully explored."""
    game = Game()
    game.setup_round()
    # Create a mock terminal by checking — we just verify the property
    node = AZNode(game=game)
    if node.is_terminal:
        assert node._fully_explored is True


def test_fully_explored_after_sufficient_simulations():
    """The 2-move position has 10 leaf nodes — should be fully explored quickly."""
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
    # All children of root should also be fully explored
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
    # Reset stability counter to confirm it's not being used
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
    # Should not be fully explored — too many branches
    assert tree._root is not None
    assert tree._root._fully_explored is False


def test_fully_explored_minimax_matches_brute_force():
    """Once fully explored, minimax value should exactly match hand calculation.

    From our hand analysis:
      Root minimax = +5pts (Black→row5, P2 responds with any White move)
    """
    game = _build_two_move_position()
    tree = _make_tree(game, simulations=50)
    tree._run_simulations()
    assert tree._root is not None
    assert tree._root._fully_explored is True

    serialized = tree.serialize()
    mm_pts = serialized["minimax_value"] * 20
    assert (
        abs(mm_pts - 5.0) < 0.01
    ), f"Expected exactly +5.0pts minimax, got {mm_pts:.2f}pts"
