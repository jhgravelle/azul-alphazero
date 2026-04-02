# neural/model.py

"""AzulNet — policy + value network for the Azul AlphaZero agent.

Architecture
------------
Input: float32 vector of shape (batch, STATE_SIZE)

    Linear(STATE_SIZE → hidden_dim)
    BatchNorm1d + ReLU
    ResBlock × num_blocks
        └─ each block: Linear → BN → ReLU → Linear → BN → skip add → ReLU
    ┌── Policy head: Linear(hidden_dim → MOVE_SPACE_SIZE)   [raw logits]
    └── Value head:  Linear(hidden_dim → 64) → ReLU
                     Linear(64 → 1) → tanh                  [scalar in (-1,1)]

Softmax is NOT applied inside the network. The caller applies it after
masking illegal moves to -inf, so illegal moves get zero probability.
"""

import torch
import torch.nn as nn

from neural.encoder import STATE_SIZE, MOVE_SPACE_SIZE


class ResBlock(nn.Module):
    """A single fully-connected residual block.

    Computes F(x) + x where F is:
        Linear → BatchNorm1d → ReLU → Linear → BatchNorm1d

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
        """Apply the residual block to x."""
        return self.relu(self.net(x) + x)


class AzulNet(nn.Module):
    """Policy + value network for Azul.

    Args:
        hidden_dim: Width of the trunk and residual blocks. Default 256.
        num_blocks: Number of residual blocks in the trunk. Default 3.
    """

    def __init__(self, hidden_dim: int = 256, num_blocks: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Trunk — project input up to hidden_dim, then residual blocks
        self.stem = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])

        # Policy head — raw logits, softmax applied externally
        self.policy_head = nn.Linear(hidden_dim, MOVE_SPACE_SIZE)

        # Value head — extra hidden layer then tanh to squash to (-1, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass."""
        trunk = self.blocks(self.stem(x))
        return self.policy_head(trunk), self.value_head(trunk)
