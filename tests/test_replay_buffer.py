"""Tests for the experience replay buffer."""

import torch
import pytest

from neural.encoder import FLAT_SIZE, MOVE_SPACE_SIZE
from neural.replay import ReplayBuffer

# ── Helpers ────────────────────────────────────────────────────────────────


def make_experience() -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    encoding = torch.rand(FLAT_SIZE)
    policy = torch.rand(MOVE_SPACE_SIZE)
    value_win = float(torch.rand(1).item() * 2 - 1)
    value_diff = float(torch.rand(1).item() * 2 - 1)
    value_abs = float(torch.rand(1).item() * 2 - 1)
    return encoding, policy, value_win, value_diff, value_abs


def fill_buffer(buf: ReplayBuffer, n: int) -> None:
    for _ in range(n):
        buf.push(*make_experience())


# ── Construction ───────────────────────────────────────────────────────────


def test_replay_buffer_starts_empty():
    assert len(ReplayBuffer(capacity=100)) == 0


def test_replay_buffer_capacity_stored():
    assert ReplayBuffer(capacity=100).capacity == 100


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
    buf = ReplayBuffer(capacity=3)
    e0 = torch.zeros(FLAT_SIZE)
    e1 = torch.ones(FLAT_SIZE)
    e2 = torch.full((FLAT_SIZE,), 2.0)
    p = torch.zeros(MOVE_SPACE_SIZE)
    buf.push(e0, p, -1.0, -1.0, -1.0)
    buf.push(e1, p, 0.0, 0.0, 0.0)
    buf.push(e2, p, 1.0, 1.0, 1.0)
    # 4th push overwrites first (value_win -1.0)
    buf.push(torch.full((FLAT_SIZE,), 3.0), p, 0.5, 0.5, 0.5)
    assert len(buf) == 3
    _, _, values_win, _, _ = buf.sample(3)
    assert -1.0 not in values_win.tolist()


# ── Sample ─────────────────────────────────────────────────────────────────


def test_sample_returns_five_tensors():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    assert len(buf.sample(8)) == 5


def test_sample_encoding_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    encodings, _, _, _, _ = buf.sample(8)
    assert encodings.shape == (8, FLAT_SIZE)


def test_sample_policy_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    _, policies, _, _, _ = buf.sample(8)
    assert policies.shape == (8, MOVE_SPACE_SIZE)


def test_sample_value_win_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    _, _, values_win, _, _ = buf.sample(8)
    assert values_win.shape == (8, 1)


def test_sample_value_diff_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    _, _, _, values_diff, _ = buf.sample(8)
    assert values_diff.shape == (8, 1)


def test_sample_value_abs_shape():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    _, _, _, _, values_abs = buf.sample(8)
    assert values_abs.shape == (8, 1)


def test_sample_dtypes_are_float32():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    sample = buf.sample(8)
    for t in sample:
        assert t.dtype == torch.float32


def test_sample_raises_when_too_few_experiences():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 5)
    with pytest.raises(ValueError):
        buf.sample(10)


def test_sample_batch_is_random():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 50)
    e1, _, _, _, _ = buf.sample(10)
    e2, _, _, _, _ = buf.sample(10)
    assert not torch.equal(e1, e2)


# ── Value target independence ──────────────────────────────────────────────


def test_sample_preserves_value_independence():
    """The three value targets are stored independently — pushing distinct
    values produces distinct tensors on sample."""
    buf = ReplayBuffer(capacity=10)
    e = torch.zeros(FLAT_SIZE)
    p = torch.zeros(MOVE_SPACE_SIZE)
    buf.push(e, p, value_win=0.1, value_diff=0.2, value_abs=0.3)
    _, _, vw, vd, va = buf.sample(1)
    assert vw.item() == pytest.approx(0.1)
    assert vd.item() == pytest.approx(0.2)
    assert va.item() == pytest.approx(0.3)


# ── Clear ──────────────────────────────────────────────────────────────────


def test_clear_empties_buffer():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    buf.clear()
    assert len(buf) == 0


def test_clear_allows_new_pushes():
    buf = ReplayBuffer(capacity=100)
    fill_buffer(buf, 20)
    buf.clear()
    buf.push(*make_experience())
    assert len(buf) == 1


def test_clear_resets_position():
    """After clear, new pushes start at position 0 (overwrite from the beginning)."""
    buf = ReplayBuffer(capacity=3)
    e0 = torch.zeros(FLAT_SIZE)
    p = torch.zeros(MOVE_SPACE_SIZE)
    buf.push(e0, p, 0.0, 0.0, 0.0)
    buf.clear()
    # A fresh push should be retrievable as the only item
    e_marker = torch.full((FLAT_SIZE,), 7.0)
    buf.push(e_marker, p, 0.9, 0.8, 0.7)
    encodings, _, _, _, _ = buf.sample(1)
    assert torch.equal(encodings[0], e_marker)
