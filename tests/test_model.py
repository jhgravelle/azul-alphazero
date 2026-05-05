# tests/test_model.py
"""Tests for the AzulNet policy + value network."""

import torch

from neural.encoder import FLAT_SIZE
from neural.model import AzulNet, ResBlock

# ── Helpers ────────────────────────────────────────────────────────────


def make_net() -> AzulNet:
    return AzulNet()


def random_input(batch: int = 1) -> torch.Tensor:
    """Return random encoding tensor (batch, FLAT_SIZE)."""
    return torch.rand(batch, FLAT_SIZE)


def forward(net: AzulNet, batch: int = 1):
    """Return ((src, tile, dst), win, diff, abs) from net."""
    return net(random_input(batch=batch))


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


# ── Forward pass: output structure ────────────────────────────────────


def test_forward_returns_policy_tuple_and_three_value_tensors():
    """forward returns ((src, tile, dst), win, diff, abs)."""
    policy_heads, v_win, v_diff, v_abs = forward(make_net())
    assert len(policy_heads) == 3
    assert isinstance(v_win, torch.Tensor)
    assert isinstance(v_diff, torch.Tensor)
    assert isinstance(v_abs, torch.Tensor)


def test_source_head_shape_single():
    (src, _, _), _, _, _ = forward(make_net(), batch=1)
    assert src.shape == (1, 2)


def test_tile_head_shape_single():
    (_, tile, _), _, _, _ = forward(make_net(), batch=1)
    assert tile.shape == (1, 5)


def test_destination_head_shape_single():
    (_, _, dst), _, _, _ = forward(make_net(), batch=1)
    assert dst.shape == (1, 6)


def test_policy_head_shapes_batched():
    (src, tile, dst), _, _, _ = forward(make_net(), batch=8)
    assert src.shape == (8, 2)
    assert tile.shape == (8, 5)
    assert dst.shape == (8, 6)


def test_value_win_shape_single():
    _, value_win, _, _ = forward(make_net(), batch=1)
    assert value_win.shape == (1, 1)


def test_value_diff_shape_single():
    _, _, value_diff, _ = forward(make_net(), batch=1)
    assert value_diff.shape == (1, 1)


def test_value_abs_shape_single():
    _, _, _, value_abs = forward(make_net(), batch=1)
    assert value_abs.shape == (1, 1)


def test_value_heads_shapes_batched():
    _, value_win, value_diff, value_abs = forward(make_net(), batch=8)
    assert value_win.shape == (8, 1)
    assert value_diff.shape == (8, 1)
    assert value_abs.shape == (8, 1)


# ── Forward pass: value range ──────────────────────────────────────────


def test_value_win_in_range():
    _, value_win, _, _ = forward(make_net(), batch=64)
    assert value_win.min().item() >= -1.0
    assert value_win.max().item() <= 1.0


def test_value_diff_in_range():
    _, _, value_diff, _ = forward(make_net(), batch=64)
    assert value_diff.min().item() >= -1.0
    assert value_diff.max().item() <= 1.0


def test_value_abs_in_range():
    _, _, _, value_abs = forward(make_net(), batch=64)
    assert value_abs.min().item() >= -1.0
    assert value_abs.max().item() <= 1.0


# ── Multi-head independence ────────────────────────────────────────────


def test_value_heads_are_distinct_modules():
    """Each value head must be its own Module, not aliased to a single one."""
    net = make_net()
    assert net.value_win_head is not net.value_diff_head
    assert net.value_win_head is not net.value_abs_head
    assert net.value_diff_head is not net.value_abs_head


def test_policy_heads_are_distinct_modules():
    net = make_net()
    assert net.source_head is not net.tile_head
    assert net.source_head is not net.destination_head
    assert net.tile_head is not net.destination_head


def test_value_heads_produce_different_outputs():
    """Three heads should compute different values on same input (fresh init)."""
    _, v_win, v_diff, v_abs = forward(make_net(), batch=32)
    assert not torch.allclose(v_win, v_diff)
    assert not torch.allclose(v_win, v_abs)
    assert not torch.allclose(v_diff, v_abs)


# ── Forward pass: policy ───────────────────────────────────────────────


def test_policy_heads_return_finite_logits():
    """Network returns raw logits — check they're finite for all three heads."""
    (src, tile, dst), _, _, _ = forward(make_net(), batch=4)
    assert torch.isfinite(src).all()
    assert torch.isfinite(tile).all()
    assert torch.isfinite(dst).all()


def test_policy_heads_softmax_to_valid_distributions():
    (src, tile, dst), _, _, _ = forward(make_net(), batch=4)
    for logits, size in [(src, 2), (tile, 5), (dst, 6)]:
        probs = torch.softmax(logits, dim=-1)
        assert probs.min().item() >= 0.0
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert probs.shape[-1] == size


# ── Gradient flow ──────────────────────────────────────────────────────


def test_gradients_flow_to_all_parameters():
    net = make_net()
    (src, tile, dst), v_win, v_diff, v_abs = net(random_input(batch=4))
    (
        src.sum() + tile.sum() + dst.sum() + v_win.sum() + v_diff.sum() + v_abs.sum()
    ).backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"


# ── Hyperparameter overrides ───────────────────────────────────────────


def test_custom_hidden_dim():
    (src, tile, dst), v_win, v_diff, v_abs = AzulNet(hidden_dim=128)(
        random_input(batch=2)
    )
    assert src.shape == (2, 2)
    assert tile.shape == (2, 5)
    assert dst.shape == (2, 6)
    assert v_win.shape == (2, 1)
    assert v_diff.shape == (2, 1)
    assert v_abs.shape == (2, 1)


def test_custom_num_blocks():
    (src, tile, dst), v_win, v_diff, v_abs = AzulNet(num_blocks=1)(
        random_input(batch=2)
    )
    assert src.shape == (2, 2)
    assert tile.shape == (2, 5)
    assert dst.shape == (2, 6)
    assert v_win.shape == (2, 1)
    assert v_diff.shape == (2, 1)
    assert v_abs.shape == (2, 1)
