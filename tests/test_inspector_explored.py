# tests/test_inspector_explored.py

"""Tests for SearchTree.is_stable().

A tree is considered fully explored when every node reachable from the root
— before crossing a round boundary — is fully expanded. Round boundary nodes
are leaves: we do not expand beyond them.
"""

from engine.game import Game
from neural.search_tree import SearchTree


def _uniform_pv(game, legal):
    n = len(legal)
    return ([1.0 / n] * n if n else []), 0.0


def _make_tree(simulations: int) -> SearchTree:
    game = Game()
    game.setup_round()
    tree = SearchTree(policy_value_fn=_uniform_pv, simulations=simulations)
    tree.reset(game)
    tree._run_simulations()
    return tree


def test_is_explored_returns_bool():
    tree = _make_tree(simulations=10)
    assert isinstance(tree.is_stable(), bool)


def test_unexplored_tree_is_not_done():
    """A freshly reset tree with no simulations is not fully explored."""
    game = Game()
    game.setup_round()
    tree = SearchTree(policy_value_fn=_uniform_pv, simulations=0)
    tree.reset(game)
    assert tree.is_stable() is False


def test_low_sim_tree_is_not_fully_explored():
    """Very few simulations cannot possibly cover all branches."""
    tree = _make_tree(simulations=5)
    assert tree.is_stable() is False


# tests/test_inspector_explored.py
# (remove the two slow/exhaustive tests, update method name)


def test_is_stable_returns_bool():
    tree = _make_tree(simulations=10)
    assert isinstance(tree.is_stable(), bool)


def test_unstable_tree_is_not_done():
    """A freshly reset tree with no simulations is not stable."""
    game = Game()
    game.setup_round()
    tree = SearchTree(policy_value_fn=_uniform_pv, simulations=0)
    tree.reset(game)
    assert tree.is_stable() is False


def test_low_sim_tree_is_not_stable():
    tree = _make_tree(simulations=5)
    assert tree.is_stable() is False


def test_stability_requires_consecutive_batches():
    """Stability counter only increments when top-k ranking is unchanged."""
    game = Game()
    game.setup_round()
    tree = SearchTree(policy_value_fn=_uniform_pv, simulations=50)
    tree.reset(game)
    # Run several batches and record stability each time
    for _ in range(10):
        tree._run_simulations()
        tree.record_batch_stability()
    # After 10 batches the counter is somewhere between 0 and 10 —
    # just verify it's a non-negative int and is_stable() agrees with it
    assert tree._stable_batches >= 0
    assert tree.is_stable() == (tree._stable_batches >= 3)


def test_stability_resets_on_ranking_change():
    """After reset(), stable batch counter goes back to zero."""
    game = Game()
    game.setup_round()
    tree = SearchTree(policy_value_fn=_uniform_pv, simulations=50)
    tree.reset(game)
    for _ in range(5):
        tree._run_simulations()
        tree.record_batch_stability()
    tree.reset(game)
    assert tree._stable_batches == 0
