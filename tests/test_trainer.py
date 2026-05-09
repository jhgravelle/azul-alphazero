# tests/test_trainer.py
"""Tests for the AzulNet trainer — loss function and training step."""

import torch
import pytest

from neural.encoder import FLAT_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import (
    Trainer,
    compute_loss,
    collect_heuristic_parallel,
    score_differential_value,
    total_score_value,
    win_loss_value,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def make_trainer() -> Trainer:
    return Trainer(AzulNet())


def make_batch(
    batch_size: int = 16,
) -> tuple[
    torch.Tensor,  # encodings     (batch, FLAT_SIZE)
    torch.Tensor,  # policies      (batch, MOVE_SPACE_SIZE)
    torch.Tensor,  # values_win    (batch, 1)
    torch.Tensor,  # values_diff   (batch, 1)
    torch.Tensor,  # values_abs    (batch, 1)
    torch.Tensor,  # policy_masks  (batch, 1)
]:
    """Return a random (encodings, policies, v_win, v_diff, v_abs, masks) batch."""
    encodings = torch.rand(batch_size, FLAT_SIZE)
    raw = torch.rand(batch_size, MOVE_SPACE_SIZE)
    policies = raw / raw.sum(dim=-1, keepdim=True)
    values_win = torch.rand(batch_size, 1) * 2 - 1
    values_diff = torch.rand(batch_size, 1) * 2 - 1
    values_abs = torch.rand(batch_size, 1) * 2 - 1
    policy_masks = torch.ones(batch_size, 1)
    return encodings, policies, values_win, values_diff, values_abs, policy_masks


def fill_buffer(buf: ReplayBuffer, n: int) -> None:
    encodings, policies, vw, vd, va, _ = make_batch(n)
    for i in range(n):
        buf.push(
            encodings[i],
            policies[i],
            vw[i, 0].item(),
            vd[i, 0].item(),
            va[i, 0].item(),
        )


# ── compute_loss ───────────────────────────────────────────────────────────


def test_compute_loss_is_positive():
    assert compute_loss(AzulNet(), *make_batch())["total"].item() > 0.0


def test_compute_loss_is_finite():
    assert torch.isfinite(compute_loss(AzulNet(), *make_batch())["total"])


def test_compute_loss_has_gradient():
    net = AzulNet()
    compute_loss(net, *make_batch())["total"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_compute_loss_decreases_toward_perfect_value():
    net = AzulNet()
    batch = make_batch(32)
    loss_before = compute_loss(net, *batch)["total"].item()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(200):
        optimizer.zero_grad()
        compute_loss(net, *batch)["total"].backward()
        optimizer.step()
    assert compute_loss(net, *batch)["total"].item() < loss_before


# ── Trainer construction ───────────────────────────────────────────────────


def test_trainer_constructs():
    make_trainer()


def test_trainer_stores_network():
    net = AzulNet()
    assert Trainer(net).net is net


def test_trainer_default_lr():
    assert make_trainer().lr == pytest.approx(1e-3)


def test_trainer_default_batch_size():
    assert make_trainer().batch_size == 256


def test_trainer_has_optimizer():
    assert make_trainer().optimizer is not None


# ── Trainer.train_step ─────────────────────────────────────────────────────


def test_train_step_loss_is_positive():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    assert trainer.train_step(buf)["total"] > 0.0


def test_train_step_updates_weights():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    before = [p.clone() for p in trainer.net.parameters()]
    trainer.train_step(buf)
    after = list(trainer.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_train_step_loss_trends_down():
    trainer = Trainer(AzulNet(), lr=1e-2)
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    losses = [trainer.train_step(buf)["total"] for _ in range(100)]
    assert sum(losses[:10]) > sum(losses[-10:])


# ── collect_heuristic_parallel ─────────────────────────────────────────────


def test_collect_heuristic_parallel_fills_buffer():
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_parallel(buf, num_pairs=2)
    assert len(buf) > 0


def test_collect_heuristic_parallel_policy_is_distribution_not_one_hot():
    """Policy targets should be multi-move distributions, not one-hot.
    At least one example in the buffer should have multiple non-zero
    entries in its policy vector."""
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_parallel(buf, num_pairs=2)
    _, policies, _, _, _, _ = buf.sample(min(len(buf), 50))
    nonzero_counts = (policies > 0).sum(dim=1)
    multi_move_rows = (nonzero_counts > 1).sum().item()
    assert multi_move_rows > 0, (
        f"Expected some examples with multi-move distributions, "
        f"got {multi_move_rows} rows with >1 non-zero entry"
    )


def test_collect_heuristic_parallel_policy_sums_to_one():
    """Every policy target with a valid mask should sum to exactly 1.0.

    Round-boundary examples have policy_mask=0.0 and a zero policy vector —
    they carry only a value target, no policy target. These are excluded.
    """
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_parallel(buf, num_pairs=2)
    _, policies, _, _, _, masks = buf.sample(min(len(buf), 30))
    valid_policies = policies[masks.squeeze(1) > 0.5]
    assert len(valid_policies) > 0, "No policy-valid examples in sample"
    sums = valid_policies.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.slow
def test_collect_heuristic_parallel_records_all_games():
    """No games should be skipped — stats should reflect 2 games per pair."""
    buf = ReplayBuffer(capacity=100_000)
    stats = collect_heuristic_parallel(buf, num_pairs=10)
    total_games = stats["wins_0"] + stats["wins_1"] + stats["ties"]
    assert total_games == 20
    assert stats["games_recorded"] == 20


# ── compute_loss — dict return and value_only ──────────────────────────────


def test_compute_loss_returns_dict():
    """compute_loss should return a dict with 'total', 'policy', 'value' keys."""
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert isinstance(result, dict)
    assert "total" in result
    assert "policy" in result
    assert "value" in result


def test_compute_loss_total_is_scalar():
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert result["total"].shape == ()


def test_compute_loss_components_are_positive():
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert result["policy"].item() > 0.0
    assert result["value"].item() > 0.0


def test_compute_loss_value_only_zeroes_policy_loss():
    """When value_only=True, policy loss must be exactly 0."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert result["policy"].item() == 0.0


def test_compute_loss_value_only_preserves_value_loss():
    """When value_only=True, value loss must still be nonzero."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert result["value"].item() > 0.0


def test_compute_loss_value_only_total_equals_value_loss():
    """When value_only=True, total == value loss (policy contributes nothing)."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert torch.isclose(result["total"], result["value"])


def test_compute_loss_value_only_gradient_flows():
    """Gradient should still flow through value head when value_only=True."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    result["total"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_compute_loss_full_has_larger_total_than_value_only():
    """Full loss (policy + value) should exceed value-only loss on a fresh net."""
    net = AzulNet()
    batch = make_batch()
    full = compute_loss(net, *batch, value_only=False)
    value_only = compute_loss(net, *batch, value_only=True)
    assert full["total"].item() > value_only["total"].item()


# ── train_step — value_only ────────────────────────────────────────────────


def test_train_step_value_only_returns_dict():
    """train_step should return a loss dict, not a plain float."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    result = trainer.train_step(buf)
    assert isinstance(result, dict)
    assert "total" in result


def test_train_step_value_only_flag_zeroes_policy_loss():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    result = trainer.train_step(buf, value_only=True)
    assert result["policy"] == 0.0


def test_train_step_value_only_still_updates_weights():
    """Even with value_only=True, the value head gradient should change weights."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    before = [p.clone() for p in trainer.net.parameters()]
    trainer.train_step(buf, value_only=True)
    after = list(trainer.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_train_step_too_small_buffer_returns_zero_dict():
    """When buffer is too small, return a zeroed dict rather than a plain 0.0."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 10)
    result = trainer.train_step(buf)
    assert isinstance(result, dict)
    assert result["total"] == 0.0
    assert result["policy"] == 0.0
    assert result["value"] == 0.0


# ── Value target functions ─────────────────────────────────────────────────


def test_win_loss_value_p0_wins():
    assert win_loss_value([50, 30], 0) == 1.0
    assert win_loss_value([50, 30], 1) == -1.0


def test_win_loss_value_p1_wins():
    assert win_loss_value([30, 50], 0) == -1.0
    assert win_loss_value([30, 50], 1) == 1.0


def test_win_loss_value_tie():
    assert win_loss_value([40, 40], 0) == 0.0
    assert win_loss_value([40, 40], 1) == 0.0


def test_score_differential_value_equal():
    assert score_differential_value([40, 40], 0) == 0.0


def test_score_differential_value_positive_boundary():
    # +50 diff / 50 divisor = +1.0 exactly
    assert score_differential_value([80, 30], 0) == pytest.approx(1.0)


def test_score_differential_value_clips_positive():
    assert score_differential_value([80, 20], 0) == 1.0


def test_score_differential_value_clips_negative():
    assert score_differential_value([20, 80], 0) == -1.0


def test_score_differential_value_midrange():
    # +25 diff / 50 = +0.5
    assert score_differential_value([60, 35], 0) == pytest.approx(0.5)


def test_total_score_value_zero():
    assert total_score_value([0, 0], 0) == 0.0


def test_total_score_value_positive_boundary():
    # score 100 / divisor 100 = +1.0
    assert total_score_value([100, 30], 0) == pytest.approx(1.0)


def test_total_score_value_clips_positive():
    assert total_score_value([110, 30], 0) == 1.0


def test_total_score_value_clips_negative():
    assert total_score_value([-110, 30], 0) == -1.0


def test_total_score_value_midrange():
    # score 40 / 100 = +0.4
    assert total_score_value([40, 30], 0) == pytest.approx(0.4)


def test_total_score_value_only_depends_on_own_score():
    """Total score for player 0 should not depend on player 1's score."""
    v1 = total_score_value([40, 10], 0)
    v2 = total_score_value([40, 80], 0)
    assert v1 == v2


# ── Multi-head value loss ──────────────────────────────────────────────────


def test_compute_loss_returns_per_head_keys():
    """compute_loss dict should include per-head breakdown."""
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert "value_win" in result
    assert "value_diff" in result
    assert "value_abs" in result


def test_compute_loss_per_head_components_are_positive():
    """Each head's MSE should be > 0 on a fresh net with random targets."""
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert result["value_win"].item() > 0.0
    assert result["value_diff"].item() > 0.0
    assert result["value_abs"].item() > 0.0


def test_compute_loss_value_equals_weighted_sum():
    """Combined value loss = _AUX_WEIGHT_WIN * win + _AUX_WEIGHT_DIFF * diff.
    value_abs is excluded from the training loss (diagnostic only)."""
    from neural.trainer import _AUX_WEIGHT_WIN, _AUX_WEIGHT_DIFF

    net = AzulNet()
    result = compute_loss(net, *make_batch())
    expected = (
        _AUX_WEIGHT_WIN * result["value_win"] + _AUX_WEIGHT_DIFF * result["value_diff"]
    )
    assert torch.isclose(result["value"], expected)


def test_compute_loss_value_only_preserves_all_three_value_losses():
    """value_only zeroes policy but all three value components remain."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert result["value_win"].item() > 0.0
    assert result["value_diff"].item() > 0.0
    assert result["value_abs"].item() > 0.0


def test_train_step_too_small_buffer_returns_all_value_keys():
    """The empty-buffer early-return dict should include all value keys."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 10)
    result = trainer.train_step(buf)
    assert result["value_win"] == 0.0
    assert result["value_diff"] == 0.0
    assert result["value_abs"] == 0.0


def test_compute_loss_value_abs_is_diagnostic_only():
    """value_abs computed but NOT included in combined value loss."""
    from neural.trainer import _AUX_WEIGHT_WIN, _AUX_WEIGHT_DIFF

    net = AzulNet()
    result = compute_loss(net, *make_batch())
    # value_abs is computed (nonzero) but absent from combined value
    assert result["value_abs"].item() > 0.0
    expected_without_abs = (
        _AUX_WEIGHT_WIN * result["value_win"] + _AUX_WEIGHT_DIFF * result["value_diff"]
    )
    assert torch.isclose(result["value"], expected_without_abs)


# ── collect_heuristic_parallel mirror pair tests ───────────────────────────


def test_collect_mirror_games_fills_buffer():
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_parallel(buf, num_pairs=2)
    assert len(buf) > 0


@pytest.mark.slow
def test_collect_mirror_games_records_double_the_pairs():
    """Each pair produces 2 games — buffer should have examples from both."""
    buf_single = ReplayBuffer(capacity=10_000)
    collect_heuristic_parallel(buf_single, num_pairs=1)

    buf_double = ReplayBuffer(capacity=10_000)
    collect_heuristic_parallel(buf_double, num_pairs=2)

    # Two pairs should produce roughly twice the examples of one pair
    assert len(buf_double) > len(buf_single)


@pytest.mark.slow
def test_collect_mirror_games_returns_correct_game_count():
    buf = ReplayBuffer(capacity=10_000)
    stats = collect_heuristic_parallel(buf, num_pairs=3)
    assert stats["games_recorded"] == 6


@pytest.mark.slow
def test_collect_mirror_games_policy_sums_to_one():
    """Every policy-valid example from mirror games should sum to 1.0.

    Round-boundary examples (policy_mask=0.0) are excluded — they have zero
    policy vectors and carry only a value target.
    """
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_parallel(buf, num_pairs=2)
    _, policies, _, _, _, masks = buf.sample(min(len(buf), 30))
    valid_policies = policies[masks.squeeze(1) > 0.5]
    assert len(valid_policies) > 0, "No policy-valid examples in sample"
    sums = valid_policies.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_mirror_games_same_seed_same_factories():
    """Two Game instances with the same seed see identical factory draws."""
    from engine.game import Game

    game_a = Game(seed=42)
    game_a.setup_round()
    factories_a = [list(f) for f in game_a.factories]

    game_b = Game(seed=42)
    game_b.setup_round()
    factories_b = [list(f) for f in game_b.factories]

    assert factories_a == factories_b


def test_mirror_games_different_seeds_different_factories():
    """Different seeds very likely produce different factory draws."""
    from engine.game import Game

    game_a = Game(seed=1)
    game_a.setup_round()
    factories_a = [list(f) for f in game_a.factories]

    game_b = Game(seed=2)
    game_b.setup_round()
    factories_b = [list(f) for f in game_b.factories]

    assert factories_a != factories_b


@pytest.mark.slow
def test_mirror_games_both_sides_represented():
    """Mirror pairs should produce wins from both p0 and p1 across enough pairs."""
    buf = ReplayBuffer(capacity=50_000)
    stats = collect_heuristic_parallel(buf, num_pairs=10)
    # With sides swapped each pair, both p0 and p1 should win some games
    assert stats["wins_0"] > 0
    assert stats["wins_1"] > 0
