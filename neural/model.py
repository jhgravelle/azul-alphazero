# neural/model.py
"""AzulNet — shrunk version for testing factory overfitting hypothesis.

Key changes from v2:
- Spatial: 8→10→32 instead of 8→64→128 (2.5× fewer spatial params)
- Multi-kernel approach: local 5×5 (10 ch) + row 1×5 (4 ch) + col 5×1 (4 ch)
  = 18 channels instead of 128
- Bottleneck: Linear(450 → 256) instead of Linear(3200 → 256) (7× fewer params)
- Overall model: ~180k params vs ~1.5M (12× smaller)

This tests whether the original model's overfitting to factory state comes from
the huge spatial bottleneck having capacity to memorize factory patterns.
"""

import torch
import torch.nn as nn

from neural.encoder import (
    FLAT_SIZE,
    MOVE_SPACE_SIZE,
    NUM_CHANNELS,
)

__all__ = ["ResBlock", "AzulNet", "MOVE_SPACE_SIZE"]


class ResBlock(nn.Module):
    """A single fully-connected residual block."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class AzulNet(nn.Module):
    """Shrunk policy + value network for Azul.

    Args:
        hidden_dim: Width of the trunk and residual blocks. Default 256.
        num_blocks: Number of residual blocks in the trunk. Default 3.
    """

    def __init__(self, hidden_dim: int = 256, num_blocks: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # ── Spatial branch — multi-kernel approach ────────────────────────
        # Input: (batch, 8, 5, 5)
        # Three parallel convolutions with different receptive fields:

        # Local 5×5 kernel: sees entire board at once, learns global patterns
        self.conv_local = nn.Conv2d(NUM_CHANNELS, 10, kernel_size=5, padding=2)
        self.norm_local = nn.LayerNorm([10, 5, 5])

        # Row kernel 1×5: sees entire rows
        self.conv_row = nn.Conv2d(NUM_CHANNELS, 4, kernel_size=(1, 5), padding=(0, 2))

        # Column kernel 5×1: sees entire columns
        self.conv_col = nn.Conv2d(NUM_CHANNELS, 4, kernel_size=(5, 1), padding=(2, 0))

        self.relu = nn.ReLU()

        # Concatenated spatial channels: 10 + 4 + 4 = 18
        spatial_flat_size = 18 * 5 * 5  # 450

        # Project spatial features to hidden_dim
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_flat_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Flat branch ────────────────────────────────────────────────────
        flat_hidden = hidden_dim // 2
        self.flat_proj = nn.Sequential(
            nn.Linear(FLAT_SIZE, flat_hidden),
            nn.LayerNorm(flat_hidden),
            nn.ReLU(),
        )

        # ── Merged trunk ───────────────────────────────────────────────────
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim + flat_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])

        # ── Heads ──────────────────────────────────────────────────────────
        self.policy_head = nn.Linear(hidden_dim, MOVE_SPACE_SIZE)
        self.value_win_head = self._make_value_head(hidden_dim)
        self.value_diff_head = self._make_value_head(hidden_dim)
        self.value_abs_head = self._make_value_head(hidden_dim)

    @staticmethod
    def _make_value_head(hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        spatial: torch.Tensor,  # (batch, 8, 5, 5)
        flat: torch.Tensor,  # (batch, FLAT_SIZE)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Returns:
            policy:     (batch, MOVE_SPACE_SIZE) raw logits
            value_win:  (batch, 1) win/loss prediction in (-1, 1)
            value_diff: (batch, 1) score-differential prediction in (-1, 1)
            value_abs:  (batch, 1) absolute-score prediction in (-1, 1)
        """
        # Spatial branch — three parallel kernels
        s_local = self.relu(
            self.norm_local(self.conv_local(spatial))
        )  # (batch, 10, 5, 5)
        s_row = self.relu(self.conv_row(spatial))  # (batch, 4, 5, 5)
        s_col = self.relu(self.conv_col(spatial))  # (batch, 4, 5, 5)

        # Concatenate all spatial features
        s = torch.cat([s_local, s_row, s_col], dim=1)  # (batch, 18, 5, 5)

        # Project to hidden dimension
        s = self.spatial_proj(s.flatten(start_dim=1))  # (batch, hidden_dim)

        # Flat branch
        f = self.flat_proj(flat)  # (batch, hidden_dim // 2)

        # Merge and trunk
        trunk = self.blocks(self.merge(torch.cat([s, f], dim=1)))

        return (
            self.policy_head(trunk),
            self.value_win_head(trunk),
            self.value_diff_head(trunk),
            self.value_abs_head(trunk),
        )
