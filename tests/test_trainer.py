# tests/test_trainer.py
"""Tests for the AzulNet trainer — loss function and training step."""

import torch
import pytest

from neural.encoder import SPATIAL_SHAPE, FLAT_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import (
    Trainer,
    compute_loss,
    collect_heuristic_games,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def make_trainer() -> Trainer:
    return Trainer(AzulNet())


def make_batch(
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random (spatials, flats, policies, values) batch."""
    spatials = torch.rand(batch_size, *SPATIAL_SHAPE)
    flats = torch.rand(batch_size, FLAT_SIZE)
    raw = torch.rand(batch_size, MOVE_SPACE_SIZE)
    policies = raw / raw.sum(dim=-1, keepdim=True)
    values = torch.rand(batch_size, 1) * 2 - 1
    return spatials, flats, policies, values


def fill_buffer(buf: ReplayBuffer, n: int) -> None:
    spatials, flats, policies, values = make_batch(n)
    for i in range(n):
        buf.push(spatials[i], flats[i], policies[i], values[i, 0].item())


# ── compute_loss ───────────────────────────────────────────────────────────


def test_compute_loss_returns_scalar():
    net = AzulNet()
    loss = compute_loss(net, *make_batch())
    assert loss.shape == ()


def test_compute_loss_is_positive():
    assert compute_loss(AzulNet(), *make_batch()).item() > 0.0


def test_compute_loss_is_finite():
    assert torch.isfinite(compute_loss(AzulNet(), *make_batch()))


def test_compute_loss_has_gradient():
    net = AzulNet()
    compute_loss(net, *make_batch()).backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_compute_loss_decreases_toward_perfect_value():
    net = AzulNet()
    batch = make_batch(32)
    loss_before = compute_loss(net, *batch).item()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(200):
        optimizer.zero_grad()
        compute_loss(net, *batch).backward()
        optimizer.step()
    assert compute_loss(net, *batch).item() < loss_before


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


def test_train_step_returns_float():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    assert isinstance(trainer.train_step(buf), float)


def test_train_step_loss_is_positive():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    assert trainer.train_step(buf) > 0.0


def test_train_step_returns_zero_when_buffer_too_small():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 10)
    assert trainer.train_step(buf) == 0.0


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
    losses = [trainer.train_step(buf) for _ in range(100)]
    assert sum(losses[:10]) > sum(losses[-10:])


# ── collect_heuristic_games ────────────────────────────────────────────────


def test_collect_heuristic_games_fills_buffer():
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_games(buf, num_games=3)
    assert len(buf) > 0


def test_collect_heuristic_games_policy_is_one_hot():
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_games(buf, num_games=2)
    _, _, policies, _ = buf.sample(min(len(buf), 10))
    sums = policies.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_collect_heuristic_games_filters_random_wins():
    buf = ReplayBuffer(capacity=100_000)
    stats = collect_heuristic_games(buf, num_games=40)
    assert stats["greedy_wins"] + stats["random_wins"] + stats["ties"] == 40
    assert stats["games_recorded"] == stats["greedy_wins"] + stats["ties"]


def test_collect_heuristic_games_values_are_score_differential():
    buf = ReplayBuffer(capacity=100_000)
    collect_heuristic_games(buf, num_games=10)
    _, _, _, values = buf.sample(min(len(buf), 50))
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    unique_values = set(values.squeeze().tolist())
    assert len(unique_values) > 3


def test_collect_self_play_warmup_records_both_players():
    from neural.model import AzulNet
    from neural.trainer import collect_self_play
    from agents.greedy import GreedyAgent

    net = AzulNet()
    buf = ReplayBuffer(capacity=100_000)
    collect_self_play(
        buf,
        net=net,
        num_games=4,
        simulations=2,
        temperature=1.0,
        opponent=GreedyAgent(),
    )
    assert len(buf) > 100, (
        f"Buffer has {len(buf)} examples from 4 games. "
        f"Expected >100 if recording both players."
    )
