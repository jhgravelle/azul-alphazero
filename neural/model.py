# neural/model.py
"""AzulNet — flat MLP architecture with v3 encoding (123 values).

v3 encoding combines all board state and game state into a single flat vector:
- Wall and pattern fills: 100 values (flattened 5×5 grids)
- Game state: 23 values (scores, penalties, tokens, inventory, bag)
- Total input: 123 values → MLP trunk → policy + value heads
"""

import torch
import torch.nn as nn

from neural.encoder import (
    FLAT_SIZE,
    MOVE_SPACE_SIZE,
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
    """Policy + value network for Azul with flat encoding.

    Args:
        hidden_dim: Width of the trunk and residual blocks. Default 256.
        num_blocks: Number of residual blocks in the trunk. Default 3.
    """

    def __init__(self, hidden_dim: int = 256, num_blocks: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # ── Input projection ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(FLAT_SIZE, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Trunk ──────────────────────────────────────────────────────────
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
        encoding: torch.Tensor,  # (batch, FLAT_SIZE=123)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            encoding: (batch, 123) tensor with board and game state

        Returns:
            policy:     (batch, MOVE_SPACE_SIZE) raw logits
            value_win:  (batch, 1) win/loss prediction in (-1, 1)
            value_diff: (batch, 1) score-differential prediction in (-1, 1)
            value_abs:  (batch, 1) absolute-score prediction in (-1, 1)
        """
        # Project input to hidden dimension and apply trunk
        trunk = self.blocks(self.input_proj(encoding))

        return (
            self.policy_head(trunk),
            self.value_win_head(trunk),
            self.value_diff_head(trunk),
            self.value_abs_head(trunk),
        )
