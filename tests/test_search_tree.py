# tests/test_search_tree.py
"""Tests for the SearchTree — MCTS with transposition table and subtree reuse."""

import pytest
from engine.game import Game, CENTER
from engine.constants import Tile, COLOR_TILES
from neural.search_tree import SearchTree, AZNode
from neural.model import AzulNet
import torch

# ── Helpers ────────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    g = Game()
    g.setup_round()
    return g


def make_policy_value_fn(net: AzulNet | None = None):
    """Return a policy/value function backed by AzulNet (or uniform if None)."""
    if net is None:

        def uniform_fn(game, legal):
            if not legal:
                return [], 0.0
            n = len(legal)
            return [1.0 / n] * n, 0.0

        return uniform_fn

    from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE
    import torch.nn.functional as F

    def net_fn(game, legal):
        spatial, flat = encode_state(game)
        spatial = spatial.unsqueeze(0)
        flat = flat.unsqueeze(0)
        net.eval()
        with torch.no_grad():
            logits, value = net(spatial, flat)
        if not legal:
            return [], value.item()
        logits = logits.squeeze(0)
        mask = torch.full((MOVE_SPACE_SIZE,), float("-inf"))
        for move in legal:
            idx = encode_move(move, game)
            mask[idx] = logits[idx]
        probs = F.softmax(mask, dim=0)
        priors = [probs[encode_move(m, game)].item() for m in legal]
        return priors, value.item()

    return net_fn


def make_tree(simulations: int = 10) -> SearchTree:
    return SearchTree(
        policy_value_fn=make_policy_value_fn(),
        simulations=simulations,
        temperature=0.0,
    )


# ── AZNode ─────────────────────────────────────────────────────────────────


def test_aznode_initial_visits_zero():
    assert AZNode(game=fresh_game()).visits == 0


def test_aznode_initial_total_value_zero():
    assert AZNode(game=fresh_game()).total_value == 0.0


def test_aznode_children_start_empty():
    assert AZNode(game=fresh_game()).children == []


def test_aznode_q_value_zero_when_unvisited():
    assert AZNode(game=fresh_game()).q_value == 0.0


def test_aznode_q_value_correct_when_visited():
    node = AZNode(game=fresh_game())
    node.visits = 4
    node.total_value = 2.0
    assert node.q_value == pytest.approx(0.5)


def test_aznode_puct_score_unvisited():
    node = AZNode(game=fresh_game(), prior=0.5)
    assert node.puct_score(parent_visits=10) > 0.0


def test_aznode_not_terminal_mid_game():
    assert not AZNode(game=fresh_game()).is_terminal


def test_aznode_round_boundary_when_all_sources_empty():
    g = fresh_game()
    for f in g.state.factories:
        f.clear()
    g.state.center.clear()
    assert AZNode(game=g).is_round_boundary


def test_aznode_not_round_boundary_with_tiles_remaining():
    g = fresh_game()
    assert not AZNode(game=g).is_round_boundary


# ── SearchTree construction ────────────────────────────────────────────────


def test_search_tree_constructs():
    make_tree()


def test_search_tree_root_none_before_reset():
    tree = make_tree()
    assert tree._root is None


def test_reset_creates_root():
    tree = make_tree()
    tree.reset(fresh_game())
    assert tree._root is not None


def test_reset_clears_transposition_table():
    tree = make_tree()
    tree.reset(fresh_game())
    tree.reset(fresh_game())
    # After second reset the table should only have the new root
    assert len(tree._table) >= 1


# ── choose_move ────────────────────────────────────────────────────────────


def test_choose_move_returns_legal_move():
    tree = make_tree(simulations=10)
    game = fresh_game()
    move = tree.choose_move(game)
    assert move in game.legal_moves()


def test_choose_move_does_not_mutate_game():
    tree = make_tree(simulations=10)
    game = fresh_game()
    factories_before = [list(f) for f in game.state.factories]
    tree.choose_move(game)
    assert [list(f) for f in game.state.factories] == factories_before


def test_choose_move_twice_returns_legal_moves():
    tree = make_tree(simulations=10)
    game = fresh_game()
    m1 = tree.choose_move(game)
    assert m1 in game.legal_moves()
    game.make_move(m1)
    tree.advance(m1)
    m2 = tree.choose_move(game)
    assert m2 in game.legal_moves()


# ── advance / subtree reuse ────────────────────────────────────────────────


def test_advance_updates_root():
    tree = make_tree(simulations=20)
    game = fresh_game()
    move = tree.choose_move(game)
    old_root = tree._root
    tree.advance(move)
    assert tree._root is not old_root


def test_advance_root_has_no_parent():
    tree = make_tree(simulations=20)
    game = fresh_game()
    move = tree.choose_move(game)
    tree.advance(move)
    assert tree._root is not None
    assert tree._root.parent is None


def test_advance_root_move_matches():
    tree = make_tree(simulations=20)
    game = fresh_game()
    move = tree.choose_move(game)
    tree.advance(move)
    assert tree._root is not None
    assert tree._root.move == move


def test_advance_preserves_visit_counts_when_child_explored():
    """If the chosen child was already explored, its visits carry forward."""
    tree = make_tree(simulations=50)
    game = fresh_game()
    move = tree.choose_move(game)
    # Find the child that matches the chosen move
    assert tree._root is not None
    chosen_child = next((c for c in tree._root.children if c.move == move), None)
    visits_before = chosen_child.visits if chosen_child else 0
    tree.advance(move)
    assert tree._root.visits == visits_before


# ── Transposition table ────────────────────────────────────────────────────


def test_transposition_table_populated_after_search():
    tree = make_tree(simulations=20)
    game = fresh_game()
    tree.choose_move(game)
    assert len(tree._table) > 1


def test_transposition_table_cleared_on_reset():
    tree = make_tree(simulations=20)
    tree.choose_move(fresh_game())
    tree.reset(fresh_game())
    # After reset only the new root should be in the table
    assert len(tree._table) == 1


# ── Factory canonicalization ───────────────────────────────────────────────


def test_canonical_moves_fewer_than_legal_when_factories_identical():
    """With two identical factories, canonical moves should be fewer."""
    tree = make_tree()
    game = Game()
    game.setup_round()
    # Force two factories to be identical
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.RED]
    game.state.factories[1] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.RED]
    legal = game.legal_moves()
    canonical = tree._canonical_moves(game)
    assert len(canonical) < len(legal)


def test_canonical_moves_equal_when_all_factories_distinct():
    """With all distinct factories, canonical moves equal legal moves."""
    tree = make_tree()
    game = Game()
    game.setup_round()
    # Force all factories to be distinct
    colors = COLOR_TILES
    game.state.factories[0] = [colors[0]] * 4
    game.state.factories[1] = [colors[1]] * 4
    game.state.factories[2] = [colors[2]] * 4
    game.state.factories[3] = [colors[3]] * 4
    game.state.factories[4] = [colors[4]] * 4
    legal = game.legal_moves()
    canonical = tree._canonical_moves(game)
    assert len(canonical) == len(legal)


def test_canonical_moves_always_includes_center_moves():
    """CENTER moves are never filtered by factory canonicalization."""
    tree = make_tree()
    game = Game()
    game.setup_round()
    game.state.center.append(Tile.BLUE)
    canonical = tree._canonical_moves(game)
    center_moves = [m for m in canonical if m.source == CENTER]
    legal_center = [m for m in game.legal_moves() if m.source == CENTER]
    assert len(center_moves) == len(legal_center)


# ── Round boundary leaf nodes ──────────────────────────────────────────────


def test_evaluate_round_boundary_returns_float():
    tree = make_tree()
    g = fresh_game()
    for f in g.state.factories:
        f.clear()
    g.state.center.clear()
    node = AZNode(game=g)
    result = tree._evaluate(node)
    assert isinstance(result, float)


def test_evaluate_round_boundary_in_range():
    tree = make_tree()
    g = fresh_game()
    for f in g.state.factories:
        f.clear()
    g.state.center.clear()
    node = AZNode(game=g)
    result = tree._evaluate(node)
    assert -1.0 <= result <= 1.0


# ── get_policy_targets ─────────────────────────────────────────────────────


def test_get_policy_targets_returns_legal_move():
    tree = make_tree(simulations=10)
    game = fresh_game()
    move, policy = tree.get_policy_targets(game)
    assert move in game.legal_moves()


def test_get_policy_targets_sums_to_one():
    tree = make_tree(simulations=10)
    game = fresh_game()
    _, policy = tree.get_policy_targets(game)
    total = sum(p for _, p in policy)
    assert total == pytest.approx(1.0, abs=0.01)


def test_get_policy_targets_all_moves_legal():
    tree = make_tree(simulations=10)
    game = fresh_game()
    _, policy = tree.get_policy_targets(game)
    legal = game.legal_moves()
    for move, _ in policy:
        assert move in legal
