# neural/model.py
"""AzulNet — flat MLP architecture with v3 encoding (123 values).

v3 encoding combines all board state and game state into a single flat vector:
- Wall and pattern fills: 100 values (flattened 5×5 grids)
- Game state: 23 values (scores, penalties, tokens, inventory, bag)
- Total input: 123 values → MLP trunk → 3-head policy + value heads

Policy is factored into three independent heads:
  source_head (2):      center vs factory
  tile_head (5):        tile color
  destination_head (6): destination (pattern lines + floor)
"""

import torch
import torch.nn as nn

from neural.encoder import FLAT_SIZE

__all__ = ["ResBlock", "AzulNet"]

HIDDEN = 64


class ResBlock(nn.Module):
    """A single fully-connected residual block."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.LayerNorm(HIDDEN),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


class AzulNet(nn.Module):
    """Policy + value network for Azul with flat encoding.

    Architecture:
        input_proj: 123 → 64 (Linear + LayerNorm + ReLU)
        trunk:      1 × ResBlock(64)
        heads:      64 → 2 / 5 / 6 (policy)
                    64 → 1 (value × 3, with Tanh)
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Input projection ───────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(FLAT_SIZE, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.ReLU(),
        )

        # ── Trunk ──────────────────────────────────────────────────────────
        self.blocks = ResBlock()

        # ── Policy heads ───────────────────────────────────────────────────
        self.source_head = nn.Linear(HIDDEN, 2)  # center vs factory
        self.tile_head = nn.Linear(HIDDEN, 5)  # tile color
        self.destination_head = nn.Linear(HIDDEN, 6)  # pattern lines + floor

        # ── Value heads ────────────────────────────────────────────────────
        self.value_win_head = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Tanh())
        self.value_diff_head = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Tanh())
        self.value_abs_head = nn.Sequential(nn.Linear(HIDDEN, 1), nn.Tanh())

    def forward(
        self,
        encoding: torch.Tensor,  # (batch, FLAT_SIZE=123)
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run a forward pass.

        Args:
            encoding: (batch, 123) tensor with board and game state

        Returns:
            policy:     3-tuple of raw logits:
                          source_logits (batch, 2)  — center vs factory
                          tile_logits   (batch, 5)  — tile color
                          dest_logits   (batch, 6)  — destination
            value_win:  (batch, 1) win/loss prediction in (-1, 1)
            value_diff: (batch, 1) score-differential prediction in (-1, 1)
            value_abs:  (batch, 1) absolute-score prediction in (-1, 1)
        """
        trunk = self.blocks(self.input_proj(encoding))

        return (
            (
                self.source_head(trunk),
                self.tile_head(trunk),
                self.destination_head(trunk),
            ),
            self.value_win_head(trunk),
            self.value_diff_head(trunk),
            self.value_abs_head(trunk),
        )
