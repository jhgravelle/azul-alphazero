# tests/test_model.py
"""Tests for the AzulNet policy + value network."""

import torch

from neural.encoder import SPATIAL_SHAPE, FLAT_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet, ResBlock

# ── Helpers ────────────────────────────────────────────────────────────────


def make_net() -> AzulNet:
    return AzulNet()


def random_input(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Return random (spatial, flat) tensors."""
    spatial = torch.rand(batch, *SPATIAL_SHAPE)
    flat = torch.rand(batch, FLAT_SIZE)
    return spatial, flat


# ── ResBlock ───────────────────────────────────────────────────────────────


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


# ── AzulNet construction ───────────────────────────────────────────────────


def test_azulnet_constructs():
    make_net()


def test_azulnet_default_hidden_dim():
    assert make_net().hidden_dim == 256


def test_azulnet_default_num_blocks():
    assert make_net().num_blocks == 3


# ── Forward pass: shapes ───────────────────────────────────────────────────


def test_forward_returns_two_tensors():
    policy, value = make_net()(*random_input())
    assert isinstance(policy, torch.Tensor)
    assert isinstance(value, torch.Tensor)


def test_policy_shape_single():
    policy, _ = make_net()(*random_input(batch=1))
    assert policy.shape == (1, MOVE_SPACE_SIZE)


def test_value_shape_single():
    _, value = make_net()(*random_input(batch=1))
    assert value.shape == (1, 1)


def test_policy_shape_batched():
    policy, _ = make_net()(*random_input(batch=8))
    assert policy.shape == (8, MOVE_SPACE_SIZE)


def test_value_shape_batched():
    _, value = make_net()(*random_input(batch=8))
    assert value.shape == (8, 1)


# ── Forward pass: value range ──────────────────────────────────────────────


def test_value_in_range():
    _, value = make_net()(*random_input(batch=64))
    assert value.min().item() >= -1.0
    assert value.max().item() <= 1.0


# ── Forward pass: policy ───────────────────────────────────────────────────


def test_policy_is_not_yet_a_distribution():
    """Network returns raw logits — just check they're finite."""
    policy, _ = make_net()(*random_input(batch=4))
    assert torch.isfinite(policy).all()


def test_policy_becomes_valid_distribution_after_softmax():
    policy, _ = make_net()(*random_input(batch=4))
    probs = torch.softmax(policy, dim=-1)
    assert probs.min().item() >= 0.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


# ── Gradient flow ──────────────────────────────────────────────────────────


def test_gradients_flow_to_all_parameters():
    net = make_net()
    policy, value = net(*random_input(batch=4))
    (policy.sum() + value.sum()).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


# ── Hyperparameter overrides ───────────────────────────────────────────────


def test_custom_hidden_dim():
    policy, value = AzulNet(hidden_dim=128)(*random_input(batch=2))
    assert policy.shape == (2, MOVE_SPACE_SIZE)
    assert value.shape == (2, 1)


def test_custom_num_blocks():
    policy, value = AzulNet(num_blocks=1)(*random_input(batch=2))
    assert policy.shape == (2, MOVE_SPACE_SIZE)
    assert value.shape == (2, 1)
