# neural/replay.py

"""Experience replay buffer for AzulNet self-play training.

Each experience stores:
    encoding     — float32 tensor of shape (FLAT_SIZE,) = (123,)
    policy       — float32 tensor of shape (MOVE_SPACE_SIZE,), MCTS visit distribution
    value_win    — float scalar in (-1, 1), win/loss outcome (primary target)
    value_diff   — float scalar in (-1, 1), normalized score differential (auxiliary)
    value_abs    — float scalar in (-1, 1), normalized absolute score (auxiliary)
    policy_mask  — float scalar, 1.0 = train policy, 0.0 = value-only (round-boundary)

The buffer is circular: once full, new experiences overwrite the oldest.
"""

import torch
from neural.encoder import FLAT_SIZE, MOVE_SPACE_SIZE


class ReplayBuffer:
    """Circular buffer storing multi-target value experiences.

    Args:
        capacity: Maximum number of experiences to store.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._encodings = torch.zeros(capacity, FLAT_SIZE, dtype=torch.float32)
        self._policies = torch.zeros(capacity, MOVE_SPACE_SIZE, dtype=torch.float32)
        self._values_win = torch.zeros(capacity, 1, dtype=torch.float32)
        self._values_diff = torch.zeros(capacity, 1, dtype=torch.float32)
        self._values_abs = torch.zeros(capacity, 1, dtype=torch.float32)
        self._policy_masks = torch.ones(capacity, 1, dtype=torch.float32)
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
        encoding: torch.Tensor,  # (FLAT_SIZE,) = (123,)
        policy: torch.Tensor,  # (MOVE_SPACE_SIZE,)
        value_win: float,
        value_diff: float,
        value_abs: float,
        policy_mask: float = 1.0,  # 0.0 for value-only (round-boundary) examples
    ) -> None:
        """Add one experience, overwriting the oldest if full."""
        self._encodings[self._pos] = encoding
        self._policies[self._pos] = policy
        self._values_win[self._pos] = value_win
        self._values_diff[self._pos] = value_diff
        self._values_abs[self._pos] = value_abs
        self._policy_masks[self._pos] = policy_mask
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[
        torch.Tensor,  # encodings     (batch, FLAT_SIZE)
        torch.Tensor,  # policies      (batch, MOVE_SPACE_SIZE)
        torch.Tensor,  # values_win    (batch, 1)
        torch.Tensor,  # values_diff   (batch, 1)
        torch.Tensor,  # values_abs    (batch, 1)
        torch.Tensor,  # policy_masks  (batch, 1)
    ]:
        """Sample a random batch without replacement.

        Returns:
            encodings:    (batch, FLAT_SIZE) — encoded game states
            policies:     (batch, MOVE_SPACE_SIZE) — MCTS visit distributions
            values_win:   (batch, 1) — win/loss targets
            values_diff:  (batch, 1) — score differential targets
            values_abs:   (batch, 1) — absolute score targets
            policy_masks: (batch, 1) — 1.0 = train policy, 0.0 = value-only

        Raises:
            ValueError: If batch_size exceeds the number of stored experiences.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer of size "
                f"{self._size}."
            )
        indices = torch.randperm(self._size)[:batch_size]
        return (
            self._encodings[indices],
            self._policies[indices],
            self._values_win[indices],
            self._values_diff[indices],
            self._values_abs[indices],
            self._policy_masks[indices],
        )
