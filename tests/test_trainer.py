# tests/test_trainer.py

"""Tests for the AzulNet trainer — loss function and training step."""

import torch
import pytest

from neural.encoder import STATE_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import Trainer, compute_loss

# ── Helpers ────────────────────────────────────────────────────────────────


def make_trainer() -> Trainer:
    """Return a Trainer with a fresh network and default hyperparameters."""
    return Trainer(AzulNet())


def make_batch(
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random (states, policies, values) batch."""
    states = torch.rand(batch_size, STATE_SIZE)
    # Policies should sum to 1 (as MCTS visit distributions do)
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
    assert loss.shape == ()  # scalar tensor


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
    # At least one parameter must have a gradient
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_compute_loss_decreases_toward_perfect_value():
    """A network predicting the correct value should have lower loss than a random
    one."""
    net = AzulNet()
    states, policies, values = make_batch(32)
    loss_before = compute_loss(net, states, policies, values).item()

    # Force the value head to output the correct values by training for a few steps
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


def test_train_step_raises_when_buffer_too_small():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 10)  # less than default batch_size of 256
    with pytest.raises(ValueError):
        trainer.train_step(buf)


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
    # Compare average of first 10 steps vs last 10 steps
    assert sum(losses[:10]) > sum(losses[-10:])
