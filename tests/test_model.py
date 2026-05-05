# tests/test_model.py
"""Tests for the AzulNet policy + value network."""

import torch

from neural.encoder import FLAT_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet, ResBlock

# ── Helpers ────────────────────────────────────────────────────────────


def make_net() -> AzulNet:
    return AzulNet()


def random_input(batch: int = 1) -> torch.Tensor:
    """Return random encoding tensor (batch, FLAT_SIZE)."""
    return torch.rand(batch, FLAT_SIZE)


# ── ResBlock ───────────────────────────────────────────────────────────


def test_resblock_output_shape():
    block = ResBlock(dim=256)
    x = torch.rand(4, 256)
    assert block(x).shape == (4, 256)


def test_resblock_is_residual():
    """With all weights zeroed, output must equal input (skip connection only)."""
    block = ResBlock(dim=256)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.rand(4, 256)
    assert torch.allclose(block(x), x)


# ── AzulNet construction ───────────────────────────────────────────────


def test_azulnet_constructs():
    make_net()


def test_azulnet_default_hidden_dim():
    assert make_net().hidden_dim == 256


def test_azulnet_default_num_blocks():
    assert make_net().num_blocks == 3


# ── Forward pass: shapes ───────────────────────────────────────────────


def test_forward_returns_four_tensors():
    """forward returns (logits, value_win, value_diff, value_abs)."""
    result = make_net()(random_input())
    assert len(result) == 4
    for t in result:
        assert isinstance(t, torch.Tensor)


def test_policy_shape_single():
    policy, _, _, _ = make_net()(random_input(batch=1))
    assert policy.shape == (1, MOVE_SPACE_SIZE)


def test_policy_shape_batched():
    policy, _, _, _ = make_net()(random_input(batch=8))
    assert policy.shape == (8, MOVE_SPACE_SIZE)


def test_value_win_shape_single():
    _, value_win, _, _ = make_net()(random_input(batch=1))
    assert value_win.shape == (1, 1)


def test_value_diff_shape_single():
    _, _, value_diff, _ = make_net()(random_input(batch=1))
    assert value_diff.shape == (1, 1)


def test_value_abs_shape_single():
    _, _, _, value_abs = make_net()(random_input(batch=1))
    assert value_abs.shape == (1, 1)


def test_value_heads_shapes_batched():
    _, value_win, value_diff, value_abs = make_net()(random_input(batch=8))
    assert value_win.shape == (8, 1)
    assert value_diff.shape == (8, 1)
    assert value_abs.shape == (8, 1)


# ── Forward pass: value range ──────────────────────────────────────────


def test_value_win_in_range():
    _, value_win, _, _ = make_net()(random_input(batch=64))
    assert value_win.min().item() >= -1.0
    assert value_win.max().item() <= 1.0


def test_value_diff_in_range():
    _, _, value_diff, _ = make_net()(random_input(batch=64))
    assert value_diff.min().item() >= -1.0
    assert value_diff.max().item() <= 1.0


def test_value_abs_in_range():
    _, _, _, value_abs = make_net()(random_input(batch=64))
    assert value_abs.min().item() >= -1.0
    assert value_abs.max().item() <= 1.0


# ── Multi-head independence ────────────────────────────────────────────


def test_value_heads_are_distinct_modules():
    """Each value head must be its own Module, not aliased to a single one."""
    net = make_net()
    assert net.value_win_head is not net.value_diff_head
    assert net.value_win_head is not net.value_abs_head
    assert net.value_diff_head is not net.value_abs_head


def test_value_heads_produce_different_outputs():
    """Three heads should compute different values on same input (fresh init)."""
    _, v_win, v_diff, v_abs = make_net()(random_input(batch=32))
    # With random initialization, it's astronomically unlikely that any two
    # heads produce identical outputs across 32 samples. If they do, the
    # heads are probably aliased to the same module.
    assert not torch.allclose(v_win, v_diff)
    assert not torch.allclose(v_win, v_abs)
    assert not torch.allclose(v_diff, v_abs)


# ── Forward pass: policy ───────────────────────────────────────────────


def test_policy_is_not_yet_a_distribution():
    """Network returns raw logits — just check they're finite."""
    policy, _, _, _ = make_net()(random_input(batch=4))
    assert torch.isfinite(policy).all()


def test_policy_becomes_valid_distribution_after_softmax():
    policy, _, _, _ = make_net()(random_input(batch=4))
    probs = torch.softmax(policy, dim=-1)
    assert probs.min().item() >= 0.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


# ── Gradient flow ──────────────────────────────────────────────────────


def test_gradients_flow_to_all_parameters():
    net = make_net()
    policy, v_win, v_diff, v_abs = net(random_input(batch=4))
    (policy.sum() + v_win.sum() + v_diff.sum() + v_abs.sum()).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


# ── Hyperparameter overrides ───────────────────────────────────────────


def test_custom_hidden_dim():
    policy, v_win, v_diff, v_abs = AzulNet(hidden_dim=128)(random_input(batch=2))
    assert policy.shape == (2, MOVE_SPACE_SIZE)
    assert v_win.shape == (2, 1)
    assert v_diff.shape == (2, 1)
    assert v_abs.shape == (2, 1)


def test_custom_num_blocks():
    policy, v_win, v_diff, v_abs = AzulNet(num_blocks=1)(random_input(batch=2))
    assert policy.shape == (2, MOVE_SPACE_SIZE)
    assert v_win.shape == (2, 1)
    assert v_diff.shape == (2, 1)
    assert v_abs.shape == (2, 1)
