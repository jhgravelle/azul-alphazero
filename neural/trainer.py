# neural/trainer.py

"""Training utilities for AzulNet."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE

logger = logging.getLogger(__name__)


# ── Loss ──────────────────────────────────────────────────────────────────────


def compute_loss(
    net: AzulNet,
    states: torch.Tensor,  # (B, STATE_SIZE)
    policies: torch.Tensor,  # (B, MOVE_SPACE_SIZE)  — target distributions
    values: torch.Tensor,  # (B, 1)                — target outcomes in (-1, 1)
) -> torch.Tensor:
    """Combined policy + value loss.

    Policy loss: cross-entropy between network output and MCTS visit distribution.
    Value  loss: MSE between network output and actual game outcome.
    """
    logits, pred_values = net(states)  # (B, MOVE), (B, 1)
    log_probs = F.log_softmax(logits, dim=1)  # (B, MOVE)
    policy_loss = -(policies * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_values, values)
    return policy_loss + value_loss


# ── Trainer ───────────────────────────────────────────────────────────────────


class Trainer:
    """Wraps AzulNet with an Adam optimizer and training step."""

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
        """Sample a batch, backpropagate, and return the scalar loss."""
        if len(buf) < self.batch_size:
            return 0.0
        states, policies, values = buf.sample(self.batch_size)
        self.net.train()
        self.optimizer.zero_grad()
        loss = compute_loss(self.net, states, policies, values)
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ── Self-play data collection ─────────────────────────────────────────────────


def collect_self_play(
    buf: ReplayBuffer,
    net: AzulNet,
    num_games: int = 50,
    simulations: int = 100,
    temperature: float = 1.0,
) -> None:
    """Play num_games games of self-play and push training examples into buf.

    Each position produces one training example:
        state   — encoded game state vector (current player's POV)
        policy  — MCTS visit distribution over all moves (MOVE_SPACE_SIZE floats)
        value   — game outcome from that player's perspective (+1 win, -1 loss, 0 tie)

    The value is filled in *retroactively* once the game ends — we play the
    whole game first, record states and policies, then go back and label them.
    """
    # Import here to avoid circular import at module level
    from agents.alphazero import AlphaZeroAgent
    from engine.game import Game

    agent = AlphaZeroAgent(net, simulations=simulations, temperature=temperature)

    for game_num in range(num_games):
        game = Game()
        game.setup_round()

        # Each entry: (player_index, state_tensor, policy_tensor)
        history: list[tuple[int, torch.Tensor, torch.Tensor]] = []

        while not game.is_game_over():
            if not game.legal_moves():
                break

            current_player = game.state.current_player
            state_vec = encode_state(game)  # (STATE_SIZE,) — current player POV

            move, policy_pairs = agent.get_policy_targets(game)

            # Build dense policy target vector
            policy_vec = torch.zeros(MOVE_SPACE_SIZE)
            for m, prob in policy_pairs:
                idx = encode_move(m, game)
                policy_vec[idx] = prob

            history.append((current_player, state_vec, policy_vec))
            game.make_move(move)

        # Determine outcome for each player
        scores = [p.score for p in game.state.players]
        if scores[0] > scores[1]:
            outcomes = {0: 1.0, 1: -1.0}
        elif scores[1] > scores[0]:
            outcomes = {0: -1.0, 1: 1.0}
        else:
            outcomes = {0: 0.0, 1: 0.0}

        # Push all positions into the replay buffer
        for player_idx, state_vec, policy_vec in history:
            value = outcomes[player_idx]
            buf.push(state_vec, policy_vec, value)

        logger.info(
            "self-play game %d/%d complete — scores %s — buffer size %d",
            game_num + 1,
            num_games,
            scores,
            len(buf),
        )
