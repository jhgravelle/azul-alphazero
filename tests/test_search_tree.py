# tests/test_search_tree.py
"""Tests for the SearchTree — MCTS with transposition table and subtree reuse."""

import pytest
from engine.game import Game, CENTER
from engine.constants import Tile, COLOR_TILES
from neural.search_tree import SearchTree, AZNode, _PUCT_C
from neural.model import AzulNet
import torch
import math
from unittest.mock import MagicMock

# ── Helpers ────────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    g = Game()
    g.setup_round()
    return g


def _make_node(**kwargs) -> AZNode:
    """Create a minimal AZNode with a mock game for unit testing."""
    game = MagicMock()
    game.is_game_over.return_value = False
    game.is_round_over.return_value = False
    return AZNode(game=game, **kwargs)


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


def make_batch_fn():
    """Return a uniform batched policy/value function for testing."""

    def fn(batch):
        results = []
        for game, legal in batch:
            if not legal:
                results.append(([], 0.0))
            else:
                n = len(legal)
                results.append(([1.0 / n] * n, 0.0))
        return results

    return fn


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
    assert node.puct_score(parent_visits=10, unvisited_exploitation=0.0) > 0.0


def test_aznode_not_terminal_mid_game():
    assert not AZNode(game=fresh_game()).is_terminal


def test_aznode_round_boundary_when_all_sources_empty():
    g = fresh_game()
    for f in g.factories:
        f.clear()
    g.center.clear()
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
    factories_before = [list(f) for f in game.factories]
    tree.choose_move(game)
    assert [list(f) for f in game.factories] == factories_before


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
    game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.RED]
    game.factories[1] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.RED]
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
    game.factories[0] = [colors[0]] * 4
    game.factories[1] = [colors[1]] * 4
    game.factories[2] = [colors[2]] * 4
    game.factories[3] = [colors[3]] * 4
    game.factories[4] = [colors[4]] * 4
    legal = game.legal_moves()
    canonical = tree._canonical_moves(game)
    assert len(canonical) == len(legal)


def test_canonical_moves_always_includes_center_moves():
    """CENTER moves are never filtered by factory canonicalization."""
    tree = make_tree()
    game = Game()
    game.setup_round()
    game.center.append(Tile.BLUE)
    canonical = tree._canonical_moves(game)
    center_moves = [m for m in canonical if m.source == CENTER]
    legal_center = [m for m in game.legal_moves() if m.source == CENTER]
    assert len(center_moves) == len(legal_center)


# ── Round boundary leaf nodes ──────────────────────────────────────────────


def test_evaluate_round_boundary_returns_float():
    tree = make_tree()
    g = fresh_game()
    for f in g.factories:
        f.clear()
    g.center.clear()
    node = AZNode(game=g)
    result = tree._evaluate(node)
    assert isinstance(result, float)


def test_evaluate_round_boundary_in_range():
    tree = make_tree()
    g = fresh_game()
    for f in g.factories:
        f.clear()
    g.center.clear()
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


# ── Batched multithreaded MCTS ─────────────────────────────────────────────


def test_batched_choose_move_returns_legal_move():
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=16,
        batch_size=4,
        temperature=0.0,
    )
    move = tree.choose_move(game)
    assert move in game.legal_moves()


def test_batched_choose_move_does_not_mutate_game():
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=16,
        batch_size=4,
        temperature=0.0,
    )
    factories_before = [list(f) for f in game.factories]
    tree.choose_move(game)
    assert [list(f) for f in game.factories] == factories_before


def test_batched_get_policy_targets_sums_to_one():
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=16,
        batch_size=4,
        temperature=0.0,
    )
    _, policy = tree.get_policy_targets(game)
    total = sum(p for _, p in policy)
    assert total == pytest.approx(1.0, abs=0.01)


def test_batched_policy_targets_all_moves_legal():
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=16,
        batch_size=4,
        temperature=0.0,
    )
    _, policy = tree.get_policy_targets(game)
    legal = game.legal_moves()
    for move, _ in policy:
        assert move in legal


def test_virtual_loss_zeroed_after_batched_search():
    """After batched search completes, no node should have residual virtual loss."""
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=16,
        batch_size=4,
        temperature=0.0,
    )
    tree.choose_move(game)

    def check_no_vl(node):
        assert (
            node.virtual_loss == 0
        ), f"Node has residual virtual_loss={node.virtual_loss}"
        for child in node.children:
            check_no_vl(child)

    assert tree._root is not None
    check_no_vl(tree._root)


def test_batched_simulations_batch_larger_than_sim_count():
    """batch_size > simulations should not crash — last batch is just smaller."""
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=3,
        batch_size=16,
        temperature=0.0,
    )
    move = tree.choose_move(game)
    assert move in game.legal_moves()


def test_batched_simulations_batch_size_one():
    """batch_size=1 should behave like single-threaded (one leaf per pass)."""
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=10,
        batch_size=1,
        temperature=0.0,
    )
    move = tree.choose_move(game)
    assert move in game.legal_moves()


def test_batched_advance_updates_root():
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=16,
        batch_size=4,
        temperature=0.0,
    )
    move = tree.choose_move(game)
    old_root = tree._root
    tree.advance(move)
    assert tree._root is not old_root


def test_batched_advance_root_has_no_parent():
    game = fresh_game()
    tree = SearchTree(
        policy_value_fn=make_policy_value_fn(),
        batch_policy_value_fn=make_batch_fn(),
        simulations=16,
        batch_size=4,
        temperature=0.0,
    )
    move = tree.choose_move(game)
    tree.advance(move)
    assert tree._root is not None
    assert tree._root.parent is None


def test_policy_value_fn_uses_value_diff():
    """make_policy_value_fn should return value_diff as the scalar value."""
    from unittest.mock import patch
    import torch
    from neural.search_tree import make_policy_value_fn
    from neural.model import AzulNet
    from engine.game import Game

    net = AzulNet()
    fn = make_policy_value_fn(net)
    game = Game()
    game.setup_round()
    legal = game.legal_moves()

    sentinel_diff = 0.333

    def fake_forward(self, encoding):
        b = encoding.shape[0]
        src = torch.zeros(b, 2)
        tile = torch.zeros(b, 5)
        dst = torch.zeros(b, 6)
        return (
            (src, tile, dst),
            torch.full((b, 1), 0.111),  # value_win
            torch.full((b, 1), sentinel_diff),  # value_diff
            torch.full((b, 1), 0.777),  # value_abs
        )

    with patch.object(AzulNet, "forward", fake_forward):
        _, value = fn(game, legal)

    assert abs(value - sentinel_diff) < 1e-5, (
        f"Expected value_diff ({sentinel_diff}), got {value}. "
        f"PUCT may be using the wrong head."
    )


def test_puct_prefers_winning_child():
    """A child where the opponent did well (high total_value) should score
    LOWER for the parent than a child where the opponent did poorly."""
    from neural.search_tree import AZNode
    from engine.game import Game

    game = Game()
    game.setup_round()
    parent = AZNode(game=game, visits=10)
    # opponent did well in this subtree — parent should avoid it
    opponent_won_child = AZNode(game=game, visits=5, total_value=4.0, prior=0.5)
    # opponent did poorly — parent should prefer it
    opponent_lost_child = AZNode(game=game, visits=5, total_value=-4.0, prior=0.5)
    assert opponent_lost_child.puct_score(
        parent.visits, unvisited_exploitation=0.0
    ) > opponent_won_child.puct_score(parent.visits, unvisited_exploitation=0.0), (
        "Parent should prefer subtree where opponent did poorly (negative "
        "total_value). If this fails, PUCT is not correctly negating child Q values."
    )


def test_puct_selects_move_leading_to_positive_value():
    """Parent should prefer child where opponent did poorly (negative total_value)."""
    from neural.search_tree import AZNode
    from engine.game import Game

    game = Game()
    game.setup_round()
    parent = AZNode(game=game, visits=100)
    child_a = AZNode(game=game, visits=50, total_value=40.0, prior=0.5)  # opponent won
    child_b = AZNode(
        game=game, visits=50, total_value=-40.0, prior=0.5
    )  # opponent lost
    score_a = child_a.puct_score(parent.visits, unvisited_exploitation=0.0)
    score_b = child_b.puct_score(parent.visits, unvisited_exploitation=0.0)
    assert score_b > score_a, (
        f"Parent should prefer child_b (opponent lost, total_value=-40, "
        f"score={score_b:.3f}) over child_a (opponent won, total_value=+40, "
        f"score={score_a:.3f}). If this fails, PUCT is not correctly negating child Q "
        f"values."
    )


def test_terminal_value_positive_for_winning_player():
    """_terminal_value should return positive for the current player if they
    are winning, negative if losing."""
    from neural.search_tree import SearchTree
    from engine.game import Game

    game = Game()
    game.setup_round()

    def dummy_fn(g, moves):
        return ([1.0 / len(moves)] * len(moves) if moves else []), 0.0

    tree = SearchTree(policy_value_fn=dummy_fn, simulations=1)

    # Rig scores so current player (0) is winning
    game.players[0].score = 40
    game.players[1].score = 20
    game.players[0]._encode()
    game.players[1]._encode()
    game.current_player_index = 0

    val = tree._terminal_value(game)
    assert val > 0, f"Winning player should get positive terminal value, got {val}"

    # Flip — now current player is losing
    game.current_player_index = 1
    val = tree._terminal_value(game)
    assert val < 0, f"Losing player should get negative terminal value, got {val}"


class TestPuctScore:
    def test_unvisited_node_uses_unvisited_exploitation(self):
        """Unvisited node exploitation should be the passed parent value, not 0.0."""
        node = _make_node(prior=0.30)
        parent_visits = 4
        unvisited_exploitation = 0.50

        score = node.puct_score(parent_visits, unvisited_exploitation)
        expected_exploration = _PUCT_C * 0.30 * math.sqrt(4) / 1
        expected = 0.50 + expected_exploration
        assert score == pytest.approx(expected)

    def test_unvisited_node_zero_exploitation_gives_old_behavior(self):
        """Passing 0.0 as unvisited_exploitation reproduces the old behavior."""
        node = _make_node(prior=0.30)
        score = node.puct_score(4, 0.0)
        expected_exploration = _PUCT_C * 0.30 * math.sqrt(4) / 1
        assert score == pytest.approx(0.0 + expected_exploration)

    def test_visited_node_uses_negated_total_value(self):
        """Visited node exploitation should be -(total_value / visits)."""
        node = _make_node(prior=0.40, visits=2, total_value=-1.10)
        score = node.puct_score(4, 0.50)
        expected_exploitation = -(-1.10) / 2  # = 0.55
        expected_exploration = _PUCT_C * 0.40 * math.sqrt(4) / 3
        assert score == pytest.approx(expected_exploitation + expected_exploration)

    def test_fully_explored_returns_negative_infinity(self):
        """Fully explored nodes should never be selected."""
        node = _make_node(prior=0.40)
        node._explored = True
        assert node.puct_score(4, 0.50) == float("-inf")

    def test_negative_parent_value_suppresses_unvisited(self):
        """A pessimistic parent value should reduce unvisited node scores."""
        node_pessimistic = _make_node(prior=0.30)
        node_neutral = _make_node(prior=0.30)
        score_pessimistic = node_pessimistic.puct_score(4, -0.50)
        score_neutral = node_neutral.puct_score(4, 0.0)
        assert score_pessimistic < score_neutral

    def test_higher_prior_wins_among_unvisited(self):
        """Among unvisited nodes with same parent value, higher prior wins."""
        node_high = _make_node(prior=0.40)
        node_low = _make_node(prior=0.10)
        assert node_high.puct_score(4, 0.50) > node_low.puct_score(4, 0.50)
