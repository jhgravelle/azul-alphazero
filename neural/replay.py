"""Experience replay buffer for AzulNet self-play training.

Each experience stores:
    spatial     — float32 tensor of shape SPATIAL_SHAPE = (12, 5, 6)
    flat        — float32 tensor of shape (FLAT_SIZE,)
    policy      — float32 tensor of shape (MOVE_SPACE_SIZE,), MCTS visit distribution
    value_win   — float scalar in (-1, 1), win/loss outcome (primary target)
    value_diff  — float scalar in (-1, 1), normalized score differential (auxiliary)
    value_abs   — float scalar in (-1, 1), normalized absolute score (auxiliary)

The buffer is circular: once full, new experiences overwrite the oldest.
"""

import torch
from neural.encoder import SPATIAL_SHAPE, FLAT_SIZE, MOVE_SPACE_SIZE


class ReplayBuffer:
    """Circular buffer storing multi-target value experiences.

    Args:
        capacity: Maximum number of experiences to store.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._spatials = torch.zeros(capacity, *SPATIAL_SHAPE, dtype=torch.float32)
        self._flats = torch.zeros(capacity, FLAT_SIZE, dtype=torch.float32)
        self._policies = torch.zeros(capacity, MOVE_SPACE_SIZE, dtype=torch.float32)
        self._values_win = torch.zeros(capacity, 1, dtype=torch.float32)
        self._values_diff = torch.zeros(capacity, 1, dtype=torch.float32)
        self._values_abs = torch.zeros(capacity, 1, dtype=torch.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        """Empty the buffer. Underlying storage tensors are retained but
        their content is no longer considered valid (size = 0, position = 0)."""
        self._pos = 0
        self._size = 0

    def push(
        self,
        spatial: torch.Tensor,  # (12, 5, 6)
        flat: torch.Tensor,  # (FLAT_SIZE,)
        policy: torch.Tensor,  # (MOVE_SPACE_SIZE,)
        value_win: float,
        value_diff: float,
        value_abs: float,
    ) -> None:
        """Add one experience, overwriting the oldest if full."""
        self._spatials[self._pos] = spatial
        self._flats[self._pos] = flat
        self._policies[self._pos] = policy
        self._values_win[self._pos] = value_win
        self._values_diff[self._pos] = value_diff
        self._values_abs[self._pos] = value_abs
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Sample a random batch without replacement.

        Returns:
            spatials:     (batch, 12, 5, 6)
            flats:        (batch, FLAT_SIZE)
            policies:     (batch, MOVE_SPACE_SIZE)
            values_win:   (batch, 1)
            values_diff:  (batch, 1)
            values_abs:   (batch, 1)
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
            self._values_win[indices],
            self._values_diff[indices],
            self._values_abs[indices],
        )
