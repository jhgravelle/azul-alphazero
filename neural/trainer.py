# neural/trainer.py

"""Training utilities for AzulNet."""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

from agents.base import Agent
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE

logger = logging.getLogger(__name__)


# ── Loss ──────────────────────────────────────────────────────────────────────


def compute_loss(
    net: AzulNet,
    states: torch.Tensor,  # (B, STATE_SIZE)
    policies: torch.Tensor,  # (B, MOVE_SPACE_SIZE) — target distributions
    values: torch.Tensor,  # (B, 1)               — target outcomes in (-1, 1)
) -> torch.Tensor:
    """Combined policy + value loss."""
    logits, pred_values = net(states)
    log_probs = F.log_softmax(logits, dim=1)
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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.net = net
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def train_step(self, buf: ReplayBuffer) -> float:
        """Sample a batch, backpropagate, and return the scalar loss."""
        if len(buf) < self.batch_size:
            return 0.0
        states, policies, values = buf.sample(self.batch_size)
        states = states.to(self.device)
        policies = policies.to(self.device)
        values = values.to(self.device)
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
    opponent: Agent | None = None,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Play num_games games and push training examples into buf.

    If opponent is None, plays AlphaZero vs AlphaZero (pure self-play).
    If opponent is provided (e.g. GreedyAgent), AlphaZero plays as player 0
    in even games and player 1 in odd games, alternating sides.

    Only AlphaZero's positions are recorded as training examples.

    Returns a list of AlphaZero's scores from each game.
    """
    from agents.alphazero import AlphaZeroAgent
    from engine.game import Game

    az_agent = AlphaZeroAgent(
        net, simulations=simulations, temperature=temperature, device=device
    )
    az_scores: list[float] = []

    for game_num in range(num_games):
        game = Game()
        game.setup_round()

        if opponent is not None:
            az_player = game_num % 2
            agents: list[Agent] = (
                [az_agent, opponent] if az_player == 0 else [opponent, az_agent]
            )
        else:
            az_player = None
            agents = [az_agent, az_agent]

        history: list[tuple[int, torch.Tensor, torch.Tensor]] = []

        while not game.is_game_over():
            if not game.legal_moves():
                break

            current_player = game.state.current_player
            is_az_turn = az_player is None or current_player == az_player

            if is_az_turn:
                state_vec = encode_state(game)
                move, policy_pairs = az_agent.get_policy_targets(game)
                policy_vec = torch.zeros(MOVE_SPACE_SIZE)
                for m, prob in policy_pairs:
                    policy_vec[encode_move(m, game)] = prob
                history.append((current_player, state_vec, policy_vec))
            else:
                move = agents[current_player].choose_move(game)

            game.make_move(move)

        scores = [p.score for p in game.state.players]
        if scores[0] > scores[1]:
            outcomes = {0: 1.0, 1: -1.0}
        elif scores[1] > scores[0]:
            outcomes = {0: -1.0, 1: 1.0}
        else:
            outcomes = {0: 0.0, 1: 0.0}

        for player_idx, state_vec, policy_vec in history:
            buf.push(state_vec, policy_vec, outcomes[player_idx])

        az_score = scores[az_player] if az_player is not None else max(scores)
        az_scores.append(az_score)

        mode = "warmup" if opponent is not None else "self-play"
        opponent_name = (
            type(opponent).__name__ if opponent is not None else "AlphaZeroAgent"
        )
        az_side = f"p{az_player}" if az_player is not None else "both"
        logger.debug(
            "%s game %d/%d -- AZ(%s) vs %s -- scores %s -- az_score=%d -- buffer "
            "size %d",
            mode,
            game_num + 1,
            num_games,
            az_side,
            opponent_name,
            scores,
            az_score,
            len(buf),
        )

    return az_scores


# ── Heuristic data collection ─────────────────────────────────────────────────


def collect_heuristic_games(
    buf: ReplayBuffer,
    num_games: int = 200,
) -> None:
    """Pretrain the buffer with GreedyAgent vs RandomAgent games.

    Every game pairs GreedyAgent against RandomAgent, alternating sides.
    This gives the network a clear signal: floor-dumping (Random) loses,
    floor-avoiding pattern-line completion (Greedy) wins.

    Logs a summary of average scores and win/loss record at the end.
    """
    from agents.greedy import GreedyAgent
    from agents.random import RandomAgent
    from engine.game import Game

    greedy_wins = 0
    random_wins = 0
    ties = 0
    greedy_scores: list[float] = []
    random_scores: list[float] = []

    for game_num in range(num_games):
        game = Game()
        game.setup_round()

        greedy = GreedyAgent()
        random_agent = RandomAgent()
        if game_num % 2 == 0:
            agents: list[Agent] = [greedy, random_agent]
            greedy_is_p0 = True
        else:
            agents = [random_agent, greedy]
            greedy_is_p0 = False

        history: list[tuple[int, torch.Tensor, torch.Tensor]] = []

        while not game.is_game_over():
            if not game.legal_moves():
                break

            current_player = game.state.current_player
            state_vec = encode_state(game)
            move = agents[current_player].choose_move(game)

            policy_vec = torch.zeros(MOVE_SPACE_SIZE)
            policy_vec[encode_move(move, game)] = 1.0

            history.append((current_player, state_vec, policy_vec))
            game.make_move(move)

        scores = [p.score for p in game.state.players]
        if scores[0] > scores[1]:
            outcomes = {0: 1.0, 1: -1.0}
        elif scores[1] > scores[0]:
            outcomes = {0: -1.0, 1: 1.0}
        else:
            outcomes = {0: 0.0, 1: 0.0}

        for player_idx, state_vec, policy_vec in history:
            buf.push(state_vec, policy_vec, outcomes[player_idx])

        greedy_score = scores[0] if greedy_is_p0 else scores[1]
        random_score = scores[1] if greedy_is_p0 else scores[0]
        greedy_scores.append(greedy_score)
        random_scores.append(random_score)
        if greedy_score > random_score:
            greedy_wins += 1
        elif random_score > greedy_score:
            random_wins += 1
        else:
            ties += 1

        greedy_label = "GreedyAgent" if greedy_is_p0 else "RandomAgent"
        random_label = "RandomAgent" if greedy_is_p0 else "GreedyAgent"
        logger.debug(
            "heuristic game %d/%d -- %s vs %s -- scores %s -- buffer size %d",
            game_num + 1,
            num_games,
            greedy_label,
            random_label,
            scores,
            len(buf),
        )

    avg_greedy = sum(greedy_scores) / len(greedy_scores)
    avg_random = sum(random_scores) / len(random_scores)
    logger.debug(
        "heuristic summary -- GreedyAgent: %d W / %d L / %d T -- "
        "avg score: Greedy %.1f, Random %.1f",
        greedy_wins,
        random_wins,
        ties,
        avg_greedy,
        avg_random,
    )
