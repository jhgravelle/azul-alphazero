# neural/replay.py
"""Experience replay buffer for AzulNet self-play training.

Each experience is a triple:
    spatial — float32 tensor of shape SPATIAL_SHAPE = (12, 5, 6)
    flat    — float32 tensor of shape (FLAT_SIZE,)
    policy  — float32 tensor of shape (MOVE_SPACE_SIZE,), MCTS visit distribution
    value   — float scalar in (-1, 1), outcome for the current player

The buffer is circular: once full, new experiences overwrite the oldest.
"""

import torch

from neural.encoder import SPATIAL_SHAPE, FLAT_SIZE, MOVE_SPACE_SIZE


class ReplayBuffer:
    """Circular buffer storing (spatial, flat, policy, value) experiences.

    Args:
        capacity: Maximum number of experiences to store.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._spatials = torch.zeros(capacity, *SPATIAL_SHAPE, dtype=torch.float32)
        self._flats = torch.zeros(capacity, FLAT_SIZE, dtype=torch.float32)
        self._policies = torch.zeros(capacity, MOVE_SPACE_SIZE, dtype=torch.float32)
        self._values = torch.zeros(capacity, 1, dtype=torch.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        spatial: torch.Tensor,  # (12, 5, 6)
        flat: torch.Tensor,  # (FLAT_SIZE,)
        policy: torch.Tensor,  # (MOVE_SPACE_SIZE,)
        value: float,
    ) -> None:
        """Add one experience, overwriting the oldest if full."""
        self._spatials[self._pos] = spatial
        self._flats[self._pos] = flat
        self._policies[self._pos] = policy
        self._values[self._pos] = value
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch without replacement.

        Returns:
            spatials:  (batch, 12, 5, 6)
            flats:     (batch, FLAT_SIZE)
            policies:  (batch, MOVE_SPACE_SIZE)
            values:    (batch, 1)
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer of size "
                f"{self._size}."
            )
        indices = torch.randperm(self._size)[:batch_size]
        return (
            self._spatials[indices],
            self._flats[indices],
            self._policies[indices],
            self._values[indices],
        )
