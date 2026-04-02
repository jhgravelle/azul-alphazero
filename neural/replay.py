# neural/replay.py

"""Experience replay buffer for AzulNet self-play training.

Each experience is a triple:
    state   — float32 tensor of shape (STATE_SIZE,)
    policy  — float32 tensor of shape (MOVE_SPACE_SIZE,), MCTS visit distribution
    value   — float scalar in (-1, 1), final game outcome for the current player

The buffer is circular: once full, new experiences overwrite the oldest.
"""

import torch

from neural.encoder import STATE_SIZE, MOVE_SPACE_SIZE


class ReplayBuffer:
    """Circular buffer storing (state, policy, value) self-play experiences.

    Args:
        capacity: Maximum number of experiences to store.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._states = torch.zeros(capacity, STATE_SIZE, dtype=torch.float32)
        self._policies = torch.zeros(capacity, MOVE_SPACE_SIZE, dtype=torch.float32)
        self._values = torch.zeros(capacity, 1, dtype=torch.float32)
        self._pos = 0  # next write position
        self._size = 0  # number of valid experiences

    def __len__(self) -> int:
        """Return the number of experiences currently stored."""
        return self._size

    def push(self, state: torch.Tensor, policy: torch.Tensor, value: float) -> None:
        """Add one experience to the buffer, overwriting the oldest if full."""
        self._states[self._pos] = state
        self._policies[self._pos] = policy
        self._values[self._pos] = value
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random batch of experiences without replacement."""
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} experiences from buffer of size "
                f"{self._size}."
            )
        indices = torch.randperm(self._size)[:batch_size]
        return (
            self._states[indices],
            self._policies[indices],
            self._values[indices],
        )
