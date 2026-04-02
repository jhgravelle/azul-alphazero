# tests/test_model.py

"""Tests for the AzulNet policy + value network."""

import torch

from neural.encoder import STATE_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet, ResBlock

# ── Helpers ────────────────────────────────────────────────────────────────


def make_net() -> AzulNet:
    """Return a freshly constructed AzulNet with default hyperparameters."""
    return AzulNet()


def random_input(batch: int = 1) -> torch.Tensor:
    """Return a random float32 input tensor of shape (batch, STATE_SIZE)."""
    return torch.rand(batch, STATE_SIZE)


# ── ResBlock ───────────────────────────────────────────────────────────────


def test_resblock_output_shape():
    """ResBlock must not change the tensor shape."""
    block = ResBlock(dim=256)
    x = torch.rand(4, 256)
    assert block(x).shape == (4, 256)


def test_resblock_is_residual():
    """With all weights zeroed, output must equal input (skip connection only)."""
    block = ResBlock(dim=256)
    # Zero out all parameters so F(x) = 0, leaving only the skip connection.
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.rand(4, 256)
    out = block(x)
    assert torch.allclose(out, x)


# ── AzulNet construction ───────────────────────────────────────────────────


def test_azulnet_constructs():
    make_net()  # should not raise


def test_azulnet_default_hidden_dim():
    net = make_net()
    assert net.hidden_dim == 256


def test_azulnet_default_num_blocks():
    net = make_net()
    assert net.num_blocks == 3


# ── Forward pass: shapes ───────────────────────────────────────────────────


def test_forward_returns_two_tensors():
    policy, value = make_net()(random_input())
    assert isinstance(policy, torch.Tensor)
    assert isinstance(value, torch.Tensor)


def test_policy_shape_single():
    policy, _ = make_net()(random_input(batch=1))
    assert policy.shape == (1, MOVE_SPACE_SIZE)


def test_value_shape_single():
    _, value = make_net()(random_input(batch=1))
    assert value.shape == (1, 1)


def test_policy_shape_batched():
    policy, _ = make_net()(random_input(batch=8))
    assert policy.shape == (8, MOVE_SPACE_SIZE)


def test_value_shape_batched():
    _, value = make_net()(random_input(batch=8))
    assert value.shape == (8, 1)


# ── Forward pass: value range ──────────────────────────────────────────────


def test_value_in_range():
    """Value head uses tanh so output must be in (-1, 1)."""
    _, value = make_net()(random_input(batch=64))
    assert value.min().item() >= -1.0
    assert value.max().item() <= 1.0


# ── Forward pass: policy ───────────────────────────────────────────────────


def test_policy_is_not_yet_a_distribution():
    """The network returns raw logits, not probabilities.
    Softmax is applied externally so we can mask illegal moves first.
    Logits can be any real number — just check they're finite.
    """
    policy, _ = make_net()(random_input(batch=4))
    assert torch.isfinite(policy).all()


def test_policy_becomes_valid_distribution_after_softmax():
    policy, _ = make_net()(random_input(batch=4))
    probs = torch.softmax(policy, dim=-1)
    assert probs.min().item() >= 0.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


# ── Gradient flow ──────────────────────────────────────────────────────────


def test_gradients_flow_to_all_parameters():
    """Every parameter must receive a gradient after a backward pass."""
    net = make_net()
    policy, value = net(random_input(batch=4))
    loss = policy.sum() + value.sum()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


# ── Hyperparameter overrides ───────────────────────────────────────────────


def test_custom_hidden_dim():
    net = AzulNet(hidden_dim=128)
    policy, value = net(random_input(batch=2))
    assert policy.shape == (2, MOVE_SPACE_SIZE)
    assert value.shape == (2, 1)


def test_custom_num_blocks():
    net = AzulNet(num_blocks=1)
    policy, value = net(random_input(batch=2))
    assert policy.shape == (2, MOVE_SPACE_SIZE)
    assert value.shape == (2, 1)
