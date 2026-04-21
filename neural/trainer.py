# neural/trainer.py
"""Training utilities for AzulNet."""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

from agents.alphazero import AlphaZeroAgent
from agents.base import Agent
from engine.game import Game, FLOOR
from engine.scoring import earned_score
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE

logger = logging.getLogger(__name__)

_SCORE_DIFF_DIVISOR = 20.0
_TOTAL_SCORE_DIVISOR = 80.0
_AUX_WEIGHT_WIN = 0.3
_AUX_WEIGHT_DIFF = 1.0
_AUX_WEIGHT_ABS = 0.3


def win_loss_value(scores: list[int], player_index: int) -> float:
    """+1 if this player won, -1 if lost, 0 if tied.

    scores should be raw scores (score - clamped_points) so floor-penalty
    games are ranked correctly even when board.score is clamped at 0.
    """
    own = scores[player_index]
    opp = scores[1 - player_index]
    if own > opp:
        return 1.0
    if own < opp:
        return -1.0
    return 0.0


def score_differential_value(scores: list[int], player_index: int) -> float:
    """Normalized score-differential value target for a player.

    scores should be raw scores (score - clamped_points) so that floor
    penalties below zero carry gradient signal even when board.score is 0.
    """
    diff = (scores[player_index] - scores[1 - player_index]) / _SCORE_DIFF_DIVISOR
    return max(-1.0, min(1.0, diff))


def total_score_value(scores: list[int], player_index: int) -> float:
    """Normalized absolute-score value target for a player.

    Only this player's score matters — opponent's score is ignored.
    Divisor chosen so that competitive-skilled-play scores (~50) land
    at the +1 boundary; typical catastrophic play (~-30) maps to -0.6.

    scores should be raw scores (score - clamped_points) so floor
    penalties below zero produce appropriately negative targets.
    """
    value = scores[player_index] / _TOTAL_SCORE_DIVISOR
    return max(-1.0, min(1.0, value))


# ── Loss ──────────────────────────────────────────────────────────────────────


# Auxiliary loss weights. value_win is the primary target and carries
# weight 1.0; the auxiliary heads regularize without dominating.


def compute_loss(
    net: AzulNet,
    spatials: torch.Tensor,  # (B, 12, 5, 6)
    flats: torch.Tensor,  # (B, FLAT_SIZE)
    policies: torch.Tensor,  # (B, MOVE_SPACE_SIZE)
    values_win: torch.Tensor,  # (B, 1)
    values_diff: torch.Tensor,  # (B, 1)
    values_abs: torch.Tensor,  # (B, 1)
    value_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Combined policy + multi-head value loss.

    Returns a dict with:
        total:      policy_loss + combined_value_loss
        policy:     cross-entropy on MCTS visit distribution (0 if value_only)
        value:      combined value loss (α·value_win + β·value_diff + value_abs)
        value_win:  MSE on win/loss target (primary)
        value_diff: MSE on score-differential target (auxiliary)
        value_abs:  MSE on absolute-score target (auxiliary)
    """
    logits, pred_win, pred_diff, pred_abs = net(spatials, flats)
    log_probs = F.log_softmax(logits, dim=1)
    if value_only:
        policy_loss = torch.tensor(0.0, requires_grad=False)
    else:
        policy_loss = -(policies * log_probs).sum(dim=1).mean()
    loss_win = F.mse_loss(pred_win, values_win)
    loss_diff = F.mse_loss(pred_diff, values_diff)
    loss_abs = F.mse_loss(pred_abs, values_abs)
    combined_value = (
        _AUX_WEIGHT_WIN * loss_win
        + _AUX_WEIGHT_DIFF * loss_diff
        + _AUX_WEIGHT_ABS * loss_abs
    )
    return {
        "total": policy_loss + combined_value,
        "policy": policy_loss,
        "value": combined_value,
        "value_win": loss_win,
        "value_diff": loss_diff,
        "value_abs": loss_abs,
    }


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

    def train_step(
        self, buf: ReplayBuffer, value_only: bool = False
    ) -> dict[str, float]:
        """Sample a batch, backpropagate, and return a loss dict."""
        if len(buf) < self.batch_size:
            return {
                "total": 0.0,
                "policy": 0.0,
                "value": 0.0,
                "value_win": 0.0,
                "value_diff": 0.0,
                "value_abs": 0.0,
            }
        spatials, flats, policies, vw, vd, va = buf.sample(self.batch_size)
        spatials = spatials.to(self.device)
        flats = flats.to(self.device)
        policies = policies.to(self.device)
        vw = vw.to(self.device)
        vd = vd.to(self.device)
        va = va.to(self.device)
        self.net.train()
        self.optimizer.zero_grad()
        loss_dict = compute_loss(
            self.net, spatials, flats, policies, vw, vd, va, value_only=value_only
        )
        loss_dict["total"].backward()
        self.optimizer.step()
        return {
            k: float(v.item() if hasattr(v, "item") else v)
            for k, v in loss_dict.items()
        }


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
    """Play num_games games and push training examples into buf."""
    from engine.game import Game

    # MCTS inference is faster on CPU than GPU for this model size.
    # Create a CPU copy of the net if it's on another device.
    cpu = torch.device("cpu")
    if next(net.parameters()).device.type != "cpu":
        net_cpu = AzulNet()
        net_cpu.load_state_dict(net.state_dict())
    else:
        net_cpu = net

    az_agent = AlphaZeroAgent(
        net_cpu,
        simulations=simulations,
        temperature=temperature,
        device=cpu,
    )
    az_scores: list[float] = []

    for game_num in range(num_games):
        game = Game()
        game.setup_round()
        az_agent.reset_tree(game)

        if opponent is not None:
            az_player = game_num % 2
            agents: list[Agent] = (
                [az_agent, opponent] if az_player == 0 else [opponent, az_agent]
            )
        else:
            az_player = None
            agents = [az_agent, az_agent]

        history: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        while not game.is_game_over():
            current_player = game.state.current_player
            is_az_turn = az_player is None or current_player == az_player

            spatial, flat = encode_state(game)

            if is_az_turn:
                # assert game.legal_moves(), (
                #     f"get_policy_targets called with no legal moves -- "
                #     f"round={game.state.round}, "
                #     f"game_over={game.is_game_over()}, "
                #     f"factories={[len(f) for f in game.state.factories]}, "
                #     f"center={game.state.center}"
                # )
                move, policy_pairs = az_agent.get_policy_targets(game)
                # In warmup mode, avoid floor moves when non-floor moves exist.
                # Untrained AZ learns to floor everything — this keeps games meaningful.
                if opponent is not None and move.destination == FLOOR:
                    legal = game.legal_moves()
                    non_floor = [m for m in legal if m.destination != FLOOR]
                    if non_floor:
                        # Pick the non-floor move with the highest policy probability.
                        policy_list = list(policy_pairs)
                        move = max(
                            non_floor,
                            key=lambda m: next(
                                (prob for pm, prob in policy_list if pm == m), 0.0
                            ),
                        )
                        non_floor_pairs = [
                            (m, p) for m, p in policy_pairs if m.destination != FLOOR
                        ]
                        total = sum(p for _, p in non_floor_pairs)
                        if total > 0:
                            policy_pairs = [(m, p / total) for m, p in non_floor_pairs]
                policy_vec = torch.zeros(MOVE_SPACE_SIZE)
                for m, prob in policy_pairs:
                    policy_vec[encode_move(m, game)] = prob
            else:
                move = agents[current_player].choose_move(game)
                policy_vec = torch.zeros(MOVE_SPACE_SIZE)
                policy_vec[encode_move(move, game)] = 1.0

            history.append((current_player, spatial, flat, policy_vec))

            prev_round = game.state.round
            game.make_move(move)
            game.advance()

            if game.state.round != prev_round:
                # assert game.legal_moves(), (
                #     f"reset_tree called with no legal moves: "
                #     f"round={game.state.round}, game_over={game.is_game_over()}"
                # )
                az_agent.reset_tree(game)
            elif not game.is_game_over():
                az_agent.advance(move)

        scores = [earned_score(p) - p.clamped_points for p in game.state.players]

        for player_idx, spatial, flat, policy_vec in history:
            vw = win_loss_value(scores, player_idx)
            vd = score_differential_value(scores, player_idx)
            va = total_score_value(scores, player_idx)
            buf.push(spatial, flat, policy_vec, vw, vd, va)

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
) -> dict[str, int]:
    """Pretrain the buffer with GreedyAgent vs CautiousAgent games.

    Both agents avoid the floor when possible, so games are coherent and
    worth recording. Policy targets are soft distributions from each
    agent's policy_distribution method, not one-hot.
    """
    from agents.cautious import CautiousAgent
    from agents.greedy import GreedyAgent

    greedy_wins = 0
    cautious_wins = 0
    ties = 0
    greedy_scores: list[float] = []
    cautious_scores: list[float] = []

    for game_num in range(num_games):
        game = Game()
        game.setup_round()

        greedy = GreedyAgent()
        cautious = CautiousAgent()
        if game_num % 2 == 0:
            agents: list[Agent] = [greedy, cautious]
            greedy_is_p0 = True
        else:
            agents = [cautious, greedy]
            greedy_is_p0 = False

        history = _play_heuristic_game(game, agents)

        scores = [earned_score(p) - p.clamped_points for p in game.state.players]
        greedy_score = scores[0] if greedy_is_p0 else scores[1]
        cautious_score = scores[1] if greedy_is_p0 else scores[0]
        greedy_scores.append(greedy_score)
        cautious_scores.append(cautious_score)

        if greedy_score > cautious_score:
            greedy_wins += 1
        elif cautious_score > greedy_score:
            cautious_wins += 1
        else:
            ties += 1

        # Every game gets recorded — both agents play coherently.
        for player_idx, spatial, flat, policy_vec in history:
            vw = win_loss_value(scores, player_idx)
            vd = score_differential_value(scores, player_idx)
            va = total_score_value(scores, player_idx)
            buf.push(spatial, flat, policy_vec, vw, vd, va)

        greedy_label = "GreedyAgent" if greedy_is_p0 else "CautiousAgent"
        cautious_label = "CautiousAgent" if greedy_is_p0 else "GreedyAgent"
        logger.debug(
            f"heuristic game {game_num + 1}/{num_games} -- "
            f"{greedy_label} vs {cautious_label} -- scores {scores} -- "
            f"buffer size {len(buf)}"
        )

    avg_greedy = sum(greedy_scores) / len(greedy_scores)
    avg_cautious = sum(cautious_scores) / len(cautious_scores)
    logger.debug(
        f"heuristic summary -- GreedyAgent: {greedy_wins} W / "
        f"{cautious_wins} L / {ties} T -- "
        f"avg score: Greedy {avg_greedy:.1f}, Cautious {avg_cautious:.1f} -- "
        f"{num_games}/{num_games} games recorded"
    )

    return {
        "greedy_wins": greedy_wins,
        "cautious_wins": cautious_wins,
        "ties": ties,
        "games_recorded": num_games,
    }


def _play_heuristic_game(
    game: "Game",
    agents: list[Agent],
) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Play a single heuristic game to completion, collecting history.

    Each history entry is (player_index, spatial, flat, policy_vec)
    where policy_vec comes from the acting agent's policy_distribution.
    """
    history: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    while not game.is_game_over():
        current_player = game.state.current_player
        spatial, flat = encode_state(game)

        agent = agents[current_player]
        policy_pairs = agent.policy_distribution(game)
        policy_vec = torch.zeros(MOVE_SPACE_SIZE)
        for m, prob in policy_pairs:
            policy_vec[encode_move(m, game)] = prob

        # The distribution is what policy_distribution returns; choose_move
        # samples from it. We ask the agent to choose, not resample ourselves,
        # to keep the agent's choose_move logic authoritative.
        move = agent.choose_move(game)

        history.append((current_player, spatial, flat, policy_vec))
        game.make_move(move)
        game.advance()

    return history
