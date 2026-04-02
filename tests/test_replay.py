# tests/test_replay.py

"""Tests for the experience replay buffer."""

import torch
import pytest

from neural.encoder import STATE_SIZE, MOVE_SPACE_SIZE
from neural.replay import ReplayBuffer

# ── Helpers ────────────────────────────────────────────────────────────────


def make_experience() -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return a random (state, policy, value) triple."""
    state = torch.rand(STATE_SIZE)
    policy = torch.rand(MOVE_SPACE_SIZE)
    value = float(torch.rand(1).item() * 2 - 1)  # uniform in (-1, 1)
    return state, policy, value


def fill_buffer(buf: ReplayBuffer, n: int) -> None:
    """Push n random experiences into buf."""
    for _ in range(n):
        buf.push(*make_experience())


# ── Construction ───────────────────────────────────────────────────────────


def test_replay_buffer_starts_empty():
    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0


def test_replay_buffer_capacity_stored():
    buf = ReplayBuffer(capacity=100)
    assert buf.capacity == 100


# ── Push and length ────────────────────────────────────────────────────────


def test_push_increases_length():
    buf = ReplayBuffer(capacity=100)
    buf.push(*make_experience())
    assert len(buf) == 1


def test_push_multiple_increases_length():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 10)
    assert len(buf) == 10


def test_length_capped_at_capacity():
    buf = ReplayBuffer(capacity=10)
    fill_buffer(buf, 25)
    assert len(buf) == 10


def test_push_overwrites_oldest_when_full():
    """After overfilling, the oldest experience must be gone."""
    buf = ReplayBuffer(capacity=3)
    # Push 3 experiences with known values
    s0, p0 = torch.zeros(STATE_SIZE), torch.zeros(MOVE_SPACE_SIZE)
    s1, p1 = torch.ones(STATE_SIZE), torch.ones(MOVE_SPACE_SIZE)
    s2, p2 = torch.full((STATE_SIZE,), 2.0), torch.full((MOVE_SPACE_SIZE,), 2.0)
    buf.push(s0, p0, -1.0)
    buf.push(s1, p1, 0.0)
    buf.push(s2, p2, 1.0)
    # Push a 4th — should overwrite the first (value -1.0)
    s3, p3 = torch.full((STATE_SIZE,), 3.0), torch.full((MOVE_SPACE_SIZE,), 3.0)
    buf.push(s3, p3, 0.5)
    assert len(buf) == 3
    # Sample all — value -1.0 must not appear
    states, policies, values = buf.sample(3)
    assert -1.0 not in values.tolist()


# ── Sample ─────────────────────────────────────────────────────────────────


def test_sample_returns_three_tensors():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    result = buf.sample(8)
    assert len(result) == 3


def test_sample_state_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    states, _, _ = buf.sample(8)
    assert states.shape == (8, STATE_SIZE)


def test_sample_policy_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    _, policies, _ = buf.sample(8)
    assert policies.shape == (8, MOVE_SPACE_SIZE)


def test_sample_value_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    _, _, values = buf.sample(8)
    assert values.shape == (8, 1)


def test_sample_dtypes_are_float32():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    states, policies, values = buf.sample(8)
    assert states.dtype == torch.float32
    assert policies.dtype == torch.float32
    assert values.dtype == torch.float32


def test_sample_raises_when_too_few_experiences():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 5)
    with pytest.raises(ValueError):
        buf.sample(10)


def test_sample_batch_is_random():
    """Two samples from the same buffer should differ (in all likelihood)."""
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 50)
    s1, _, _ = buf.sample(10)
    s2, _, _ = buf.sample(10)
    assert not torch.equal(s1, s2)
