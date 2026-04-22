# tests/test_search_tree_serializer.py

"""Tests for SearchTree.serialize() — the tree inspector data structure.

serialize() walks the tree from _root after search and returns a nested dict
suitable for JSON serialization. It is the data contract between the backend
and the inspector frontend.
"""

from engine.game import Game
from neural.search_tree import SearchTree

# ── Helpers ───────────────────────────────────────────────────────────────────


def _uniform_pv(game, legal):
    """Policy/value fn: uniform priors, zero value. No net required."""
    n = len(legal)
    priors = [1.0 / n] * n if n else []
    return priors, 0.0


def _make_tree(simulations: int = 50) -> SearchTree:
    game = Game()
    game.setup_round()
    tree = SearchTree(policy_value_fn=_uniform_pv, simulations=simulations)
    tree.reset(game)
    tree._run_simulations()
    return tree


# ── Shape and types ───────────────────────────────────────────────────────────


def test_serialize_returns_dict():
    tree = _make_tree()
    result = tree.serialize()
    assert isinstance(result, dict)


def test_serialize_root_has_required_keys():
    tree = _make_tree()
    node = tree.serialize()
    for key in (
        "key",
        "move",
        "visits",
        "value_diff",
        "prior",
        "is_round_boundary",
        "depth",
        "children",
    ):
        assert key in node, f"missing key: {key!r}"


def test_serialize_root_move_is_none():
    """Root has no move from parent."""
    tree = _make_tree()
    assert tree.serialize()["move"] is None


def test_serialize_root_depth_is_zero():
    tree = _make_tree()
    assert tree.serialize()["depth"] == 0


def test_serialize_root_visits_equals_simulations():
    tree = _make_tree(simulations=50)
    assert tree.serialize()["visits"] == 50


def test_serialize_children_is_list():
    tree = _make_tree()
    assert isinstance(tree.serialize()["children"], list)


def test_serialize_children_have_depth_one():
    tree = _make_tree()
    for child in tree.serialize()["children"]:
        assert child["depth"] == 1


def test_serialize_children_depth_increments():
    """Grandchildren should be depth 2."""
    tree = _make_tree(simulations=200)
    for child in tree.serialize()["children"]:
        for grandchild in child["children"]:
            assert grandchild["depth"] == 2


def test_serialize_value_diff_is_float():
    tree = _make_tree()
    node = tree.serialize()
    assert isinstance(node["value_diff"], float)


def test_serialize_prior_is_float():
    tree = _make_tree()
    for child in tree.serialize()["children"]:
        assert isinstance(child["prior"], float)


def test_serialize_key_is_hex_string():
    tree = _make_tree()
    key = tree.serialize()["key"]
    assert isinstance(key, str)
    int(key, 16)  # raises ValueError if not valid hex


def test_serialize_child_keys_are_unique():
    tree = _make_tree(simulations=200)
    node = tree.serialize()
    keys = [c["key"] for c in node["children"]]
    assert len(keys) == len(set(keys))


# ── top_k ─────────────────────────────────────────────────────────────────────


def test_serialize_top_k_limits_children():
    tree = _make_tree(simulations=200)
    result = tree.serialize(top_k=3)
    assert len(result["children"]) <= 3


def test_serialize_children_sorted_by_cumulative_immediate():
    """Children should be sorted by cumulative_immediate descending."""
    tree = _make_tree(simulations=200)
    children = tree.serialize(top_k=5)["children"]
    cumulatives = [
        c["cumulative_immediate"]
        for c in children
        if c["cumulative_immediate"] is not None
    ]
    assert cumulatives == sorted(cumulatives, reverse=True)


def test_serialize_top_k_one_returns_best_child():
    tree = _make_tree(simulations=200)
    # Find the highest-visit child from the root node directly
    assert tree._root
    visited = [c for c in tree._root.children if c.visits > 0]
    best_node = max(visited, key=lambda c: c.visits)
    children_one = tree.serialize(top_k=1)["children"]
    assert children_one[0]["key"] == hex(best_node.zobrist_hash)


# ── max_depth ─────────────────────────────────────────────────────────────────


def test_serialize_max_depth_zero_has_no_children():
    tree = _make_tree(simulations=200)
    result = tree.serialize(max_depth=0)
    assert result["children"] == []


def test_serialize_max_depth_one_children_have_no_children():
    tree = _make_tree(simulations=200)
    result = tree.serialize(max_depth=1)
    for child in result["children"]:
        assert child["children"] == []


def test_serialize_max_depth_two_grandchildren_have_no_children():
    tree = _make_tree(simulations=200)
    result = tree.serialize(max_depth=2)
    for child in result["children"]:
        for grandchild in child["children"]:
            assert grandchild["children"] == []


# ── move string ───────────────────────────────────────────────────────────────


def test_serialize_child_move_is_string():
    tree = _make_tree()
    for child in tree.serialize()["children"]:
        assert isinstance(child["move"], str)


def test_serialize_child_move_is_nonempty():
    tree = _make_tree()
    for child in tree.serialize()["children"]:
        assert len(child["move"]) > 0


# ── is_round_boundary ────────────────────────────────────────────────────────


def test_serialize_root_is_round_boundary_is_bool():
    tree = _make_tree()
    assert isinstance(tree.serialize()["is_round_boundary"], bool)


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_serialize_before_simulations_returns_root_with_no_children():
    """A freshly reset tree with no simulations should serialize without error."""
    game = Game()
    game.setup_round()
    tree = SearchTree(policy_value_fn=_uniform_pv, simulations=0)
    tree.reset(game)
    result = tree.serialize()
    assert result["visits"] == 0
    assert result["children"] == []


def test_serialize_is_json_serializable():
    """The full output must survive json.dumps without error."""
    import json

    tree = _make_tree(simulations=100)
    result = tree.serialize(max_depth=3, top_k=5)
    json.dumps(result)  # raises TypeError if anything is non-serializable
