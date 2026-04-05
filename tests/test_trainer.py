# tests/test_trainer.py

"""Tests for the AzulNet trainer — loss function and training step."""

import torch
import pytest

from neural.encoder import STATE_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import (
    Trainer,
    compute_loss,
    collect_heuristic_games,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def make_trainer() -> Trainer:
    """Return a Trainer with a fresh network and default hyperparameters."""
    return Trainer(AzulNet())


def make_batch(
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random (states, policies, values) batch."""
    states = torch.rand(batch_size, STATE_SIZE)
    raw = torch.rand(batch_size, MOVE_SPACE_SIZE)
    policies = raw / raw.sum(dim=-1, keepdim=True)
    values = torch.rand(batch_size, 1) * 2 - 1  # uniform in (-1, 1)
    return states, policies, values


def fill_buffer(buf: ReplayBuffer, n: int) -> None:
    states, policies, values = make_batch(n)
    for i in range(n):
        buf.push(states[i], policies[i], values[i, 0].item())


# ── compute_loss ───────────────────────────────────────────────────────────


def test_compute_loss_returns_scalar():
    net = AzulNet()
    states, policies, values = make_batch()
    loss = compute_loss(net, states, policies, values)
    assert loss.shape == ()


def test_compute_loss_is_positive():
    net = AzulNet()
    states, policies, values = make_batch()
    loss = compute_loss(net, states, policies, values)
    assert loss.item() > 0.0


def test_compute_loss_is_finite():
    net = AzulNet()
    states, policies, values = make_batch()
    loss = compute_loss(net, states, policies, values)
    assert torch.isfinite(loss)


def test_compute_loss_has_gradient():
    net = AzulNet()
    states, policies, values = make_batch()
    loss = compute_loss(net, states, policies, values)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_compute_loss_decreases_toward_perfect_value():
    """A network trained on a fixed batch should have lower loss after training."""
    net = AzulNet()
    states, policies, values = make_batch(32)
    loss_before = compute_loss(net, states, policies, values).item()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(200):
        optimizer.zero_grad()
        compute_loss(net, states, policies, values).backward()
        optimizer.step()

    loss_after = compute_loss(net, states, policies, values).item()
    assert loss_after < loss_before


# ── Trainer construction ───────────────────────────────────────────────────


def test_trainer_constructs():
    make_trainer()


def test_trainer_stores_network():
    net = AzulNet()
    trainer = Trainer(net)
    assert trainer.net is net


def test_trainer_default_lr():
    trainer = make_trainer()
    assert trainer.lr == pytest.approx(1e-3)


def test_trainer_default_batch_size():
    trainer = make_trainer()
    assert trainer.batch_size == 256


def test_trainer_has_optimizer():
    trainer = make_trainer()
    assert trainer.optimizer is not None


# ── Trainer.train_step ─────────────────────────────────────────────────────


def test_train_step_returns_float():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    result = trainer.train_step(buf)
    assert isinstance(result, float)


def test_train_step_loss_is_positive():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    assert trainer.train_step(buf) > 0.0


def test_train_step_returns_zero_when_buffer_too_small():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 10)  # less than default batch_size of 256
    loss = trainer.train_step(buf)
    assert loss == 0.0


def test_train_step_updates_weights():
    """Weights must change after a training step."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)

    before = [p.clone() for p in trainer.net.parameters()]
    trainer.train_step(buf)
    after = list(trainer.net.parameters())

    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed


def test_train_step_loss_trends_down():
    """Loss should decrease over many training steps on a fixed buffer."""
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
    _, policies, _ = buf.sample(min(len(buf), 10))
    sums = policies.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_collect_heuristic_games_filters_random_wins():
    """When Random wins, that game's examples are not added to the buffer.

    We run 40 games. Random wins roughly 5-10% of the time, so we expect
    some games to be filtered. The returned stats should account for all games,
    and the buffer should only contain examples from games Greedy won or tied.
    """
    buf = ReplayBuffer(capacity=100_000)
    stats = collect_heuristic_games(buf, num_games=40)

    assert stats["greedy_wins"] + stats["random_wins"] + stats["ties"] == 40
    assert stats["games_recorded"] == stats["greedy_wins"] + stats["ties"]


def test_collect_heuristic_games_values_are_score_differential():
    """Value targets should be normalized score differential, not binary win/loss.

    Greedy typically scores 20-40 and Random scores 0-5, so the value for
    Greedy's positions should be positive but less than 1.0 (not exactly 1.0).
    """
    buf = ReplayBuffer(capacity=100_000)
    collect_heuristic_games(buf, num_games=10)

    _, _, values = buf.sample(min(len(buf), 50))

    # Values should be in [-1, 1]
    assert values.min() >= -1.0
    assert values.max() <= 1.0

    # With score differential, values should NOT all be exactly +1/-1/0.
    # Binary win/loss would give only those three values.
    unique_values = set(values.squeeze().tolist())
    assert len(unique_values) > 3, (
        f"Expected continuous values, got only {unique_values}. "
        f"Still using binary win/loss?"
    )


def test_collect_self_play_warmup_records_both_players():
    """In warmup mode, both AZ and opponent positions should be recorded.

    With only AZ positions, the buffer gets ~50% of the moves per game.
    With both players, it should get roughly twice as many examples.
    """
    from neural.model import AzulNet
    from neural.trainer import collect_self_play
    from agents.greedy import GreedyAgent

    net = AzulNet()

    # AZ-only buffer
    # buf_az_only = ReplayBuffer(capacity=100_000)
    # We need the old behavior to compare, so we'll just check the new buffer
    # has a reasonable number of examples

    buf = ReplayBuffer(capacity=100_000)
    collect_self_play(
        buf,
        net=net,
        num_games=4,
        simulations=2,
        temperature=1.0,
        opponent=GreedyAgent(),
    )

    # A typical Azul game has ~30-60 total moves (both players combined).
    # 4 games × ~40 moves = ~160 examples if recording both players.
    # If only recording AZ, we'd get ~80.
    # Use 100 as threshold — comfortably above AZ-only, below both-players.
    assert len(buf) > 100, (
        f"Buffer has {len(buf)} examples from 4 games. "
        f"Expected >100 if recording both players."
    )
