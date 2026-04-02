# neural/trainer.py

"""Trainer for AzulNet — loss function and training step.

Loss
----
Total loss = value_loss + policy_loss

    value_loss  = MSE(predicted_value, actual_outcome)
    policy_loss = cross_entropy(predicted_logits, mcts_visit_distribution)

The policy target is a probability distribution (MCTS visit counts normalized
to sum to 1). Cross-entropy against a soft target is computed as:

    policy_loss = -sum(target * log_softmax(logits), dim=-1).mean()

Self-play data collection is stubbed here and will be completed in Phase 6
once AlphaZeroAgent exists.
"""

import logging

import torch
import torch.nn.functional as F

from neural.model import AzulNet
from neural.replay import ReplayBuffer

logger = logging.getLogger(__name__)


def compute_loss(
    net: AzulNet,
    states: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """Compute the combined policy + value loss for a batch."""
    policy_logits, predicted_values = net(states)

    value_loss = F.mse_loss(predicted_values, values)
    policy_loss = -(policies * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

    return value_loss + policy_loss


class Trainer:
    """Manages network training from replay buffer experiences.

    Args:
        net:        The AzulNet instance to train.
        lr:         Learning rate for the Adam optimizer. Default 1e-3.
        batch_size: Number of experiences per training step. Default 256.
    """

    def __init__(
        self,
        net: AzulNet,
        lr: float = 1e-3,
        batch_size: int = 256,
    ) -> None:
        self.net = net
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def train_step(self, buf: ReplayBuffer) -> float:
        """Sample one batch from the buffer and take one gradient step."""
        states, policies, values = buf.sample(self.batch_size)
        self.optimizer.zero_grad()
        loss = compute_loss(self.net, states, policies, values)
        loss.backward()
        self.optimizer.step()
        logger.debug("train_step loss=%.4f", loss.item())
        return loss.item()

    def collect_self_play(self, buf: ReplayBuffer, num_games: int) -> None:
        """Play num_games games and push experiences into buf.

        Stub — will be implemented in Phase 6 once AlphaZeroAgent exists.

        Args:
            buf:       The replay buffer to fill.
            num_games: Number of self-play games to run.
        """
        raise NotImplementedError
