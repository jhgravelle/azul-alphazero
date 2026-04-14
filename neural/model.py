# neural/model.py
"""AzulNet — policy + value network for the Azul AlphaZero agent.

Architecture
------------
Two input branches that merge before the policy and value heads.

Spatial branch — processes the (12, 5, 6) wall+pattern tensor:
    Conv2d(12 → 64, kernel=3, padding=1) → LayerNorm → ReLU
    Conv2d(64 → 128, kernel=3, padding=1) → LayerNorm → ReLU
    Flatten → Linear(128*5*6 → hidden_dim) → LayerNorm → ReLU

Flat branch — processes the (FLAT_SIZE,) feature vector:
    Linear(FLAT_SIZE → hidden_dim // 2) → LayerNorm → ReLU

Merged trunk:
    Linear(hidden_dim + hidden_dim // 2 → hidden_dim) → LayerNorm → ReLU
    ResBlock × num_blocks

Policy head:
    Linear(hidden_dim → MOVE_SPACE_SIZE)   [raw logits]

Value head:
    Linear(hidden_dim → 64) → ReLU
    Linear(64 → 1) → Tanh                  [scalar in (-1, 1)]

Softmax is NOT applied inside the network. The caller applies it after
masking illegal moves to -inf so illegal moves get zero probability.
"""

import torch
import torch.nn as nn

from neural.encoder import (
    FLAT_SIZE,
    MOVE_SPACE_SIZE,
    NUM_CHANNELS,
    BOARD_SIZE,
    GRID_COLS,
)

# Re-export so callers that do `from neural.model import MOVE_SPACE_SIZE` still work.
__all__ = ["ResBlock", "AzulNet", "MOVE_SPACE_SIZE"]

_CONV_MID = 64
_CONV_OUT = 128


class ResBlock(nn.Module):
    """A single fully-connected residual block.

    Computes F(x) + x where F is:
        Linear → LayerNorm → ReLU → Linear → LayerNorm

    Args:
        dim: Width of the block (input and output size are identical).
    """

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
    """Policy + value network for Azul.

    Args:
        hidden_dim: Width of the trunk and residual blocks. Default 256.
        num_blocks:  Number of residual blocks in the trunk. Default 3.
    """

    def __init__(self, hidden_dim: int = 256, num_blocks: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # ── Spatial branch ────────────────────────────────────────────────
        # Input: (batch, 12, 5, 6)
        # Two conv layers preserve spatial dimensions (padding=1, kernel=3).
        # LayerNorm applied over the channel dimension after each conv.
        self.conv1 = nn.Conv2d(NUM_CHANNELS, _CONV_MID, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm([_CONV_MID, BOARD_SIZE, GRID_COLS])
        self.conv2 = nn.Conv2d(_CONV_MID, _CONV_OUT, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm([_CONV_OUT, BOARD_SIZE, GRID_COLS])
        self.relu = nn.ReLU()

        conv_flat_size = _CONV_OUT * BOARD_SIZE * GRID_COLS  # 128 * 5 * 6 = 3840
        self.spatial_proj = nn.Sequential(
            nn.Linear(conv_flat_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Flat branch ───────────────────────────────────────────────────
        flat_hidden = hidden_dim // 2
        self.flat_proj = nn.Sequential(
            nn.Linear(FLAT_SIZE, flat_hidden),
            nn.LayerNorm(flat_hidden),
            nn.ReLU(),
        )

        # ── Merged trunk ──────────────────────────────────────────────────
        self.merge = nn.Sequential(
            nn.Linear(hidden_dim + flat_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])

        # ── Heads ─────────────────────────────────────────────────────────
        self.policy_head = nn.Linear(hidden_dim, MOVE_SPACE_SIZE)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        spatial: torch.Tensor,  # (batch, 12, 5, 6)
        flat: torch.Tensor,  # (batch, FLAT_SIZE)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Returns:
            policy: (batch, MOVE_SPACE_SIZE) raw logits
            value:  (batch, 1) scalar in (-1, 1)
        """
        # Spatial branch
        s = self.relu(self.norm1(self.conv1(spatial)))
        s = self.relu(self.norm2(self.conv2(s)))
        s = self.spatial_proj(s.flatten(start_dim=1))

        # Flat branch
        f = self.flat_proj(flat)

        # Merge and trunk
        trunk = self.blocks(self.merge(torch.cat([s, f], dim=1)))
        return self.policy_head(trunk), self.value_head(trunk)
