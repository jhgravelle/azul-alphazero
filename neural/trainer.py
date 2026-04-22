# neural/trainer.py
"""Training utilities for AzulNet."""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
from typing import Callable
import random
from agents.alphazero import AlphaZeroAgent
from agents.base import Agent
from engine.game import Game, FLOOR
from engine.scoring import earned_score_unclamped
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE

logger = logging.getLogger(__name__)

_SCORE_DIFF_DIVISOR = 50.0  # matches encoder SCORE_DELTA_DIVISOR
_TOTAL_SCORE_DIVISOR = 80.0
_AUX_WEIGHT_WIN = 0.3
_AUX_WEIGHT_DIFF = 1.0
_AUX_WEIGHT_ABS = 0.1  # reduced from 0.3; candidate for removal


def win_loss_value(scores: list[int], player_index: int) -> float:
    """+1 if this player won, -1 if lost, 0 if tied.

    scores should be earned_score_unclamped values so floor-penalty games
    are ranked correctly even when board.score is clamped at 0.
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

    Uses earned_score_unclamped values so that floor penalties below zero
    carry gradient signal even when board.score is 0.
    """
    diff = (scores[player_index] - scores[1 - player_index]) / _SCORE_DIFF_DIVISOR
    return max(-1.0, min(1.0, diff))


def total_score_value(scores: list[int], player_index: int) -> float:
    """Normalized absolute-score value target for a player.

    Only this player's score matters — opponent's score is ignored.
    Divisor chosen so that competitive-skilled-play scores (~50) land
    at the +1 boundary; typical catastrophic play (~-30) maps to -0.6.

    Uses earned_score_unclamped values so floor penalties below zero
    produce appropriately negative targets.
    """
    value = scores[player_index] / _TOTAL_SCORE_DIVISOR
    return max(-1.0, min(1.0, value))


# ── Loss ──────────────────────────────────────────────────────────────────────


def compute_loss(
    net: AzulNet,
    spatials: torch.Tensor,  # (B, 14, 5, 6)
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
        value:      combined value loss
        value_win:  MSE on win/loss target (primary)
        value_diff: MSE on score-differential target (auxiliary)
        value_abs:  MSE on absolute-score target (auxiliary, weight 0.1)
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


def _compute_game_scores(game: Game) -> list[int]:
    """Return earned_score_unclamped minus clamped_points for each player.

    This is the correct scoring signal: unclamped so floor penalties below
    zero carry meaning, minus clamped_points so score-clamping artifacts
    don't inflate the target.
    """
    return [
        earned_score_unclamped(player) - player.clamped_points
        for player in game.state.players
    ]


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
                move, policy_pairs = az_agent.get_policy_targets(game)
                # In warmup mode, avoid floor moves when non-floor moves exist.
                # Untrained AZ learns to floor everything — this keeps games meaningful.
                if opponent is not None and move.destination == FLOOR:
                    legal = game.legal_moves()
                    non_floor = [m for m in legal if m.destination != FLOOR]
                    if non_floor:
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
                az_agent.reset_tree(game)
            elif not game.is_game_over():
                az_agent.advance(move)

        scores = _compute_game_scores(game)

        for player_idx, spatial, flat, policy_vec in history:
            vw = win_loss_value(scores, player_idx)
            vd = score_differential_value(scores, player_idx)
            va = total_score_value(scores, player_idx)
            buf.push(spatial, flat, policy_vec, vw, vd, va)

        az_score = scores[az_player] if az_player is not None else max(scores)
        az_scores.append(az_score)

        # mode = "warmup" if opponent is not None else "self-play"
        opponent_name = (
            type(opponent).__name__ if opponent is not None else "AlphaZeroAgent"
        )
        az_side = f"p{az_player}" if az_player is not None else "both"
        logger.debug(
            f"{'warmup' if opponent is not None else 'self-play'} game "
            f"{game_num + 1}/{num_games} -- AZ({az_side}) vs {opponent_name} -- "
            f"scores {scores} -- az_score={az_score} -- buffer size {len(buf)}"
        )

    return az_scores


# ── Heuristic data collection ─────────────────────────────────────────────────

# Type alias for a matchup: two agent factories and a relative weight.
# The weight controls how often this matchup is sampled relative to others.
# Example:
#   MATCHUPS_DEFAULT = [
#       (make_easy, make_easy,   0.3),
#       (make_easy, make_medium, 0.4),
#       (make_medium, make_medium, 0.3),
#   ]

MatchupSpec = tuple[Callable, Callable, float]


def _default_matchups() -> list[MatchupSpec]:
    """Default weighted matchup list for heuristic pretraining.

    easy vs easy:     30% — high volume, lower quality ceiling
    easy vs medium:   40% — most useful, covers both skill levels
    medium vs medium: 30% — fewer games, higher quality ceiling
    """
    from agents.alphabeta import AlphaBetaAgent

    def make_easy() -> AlphaBetaAgent:
        return AlphaBetaAgent(depths=(2, 3, 7), thresholds=(20, 10))

    def make_medium() -> AlphaBetaAgent:
        return AlphaBetaAgent(depths=(3, 5, 7), thresholds=(20, 10))

    return [
        (make_easy, make_easy, 0.3),
        (make_easy, make_medium, 0.4),
        (make_medium, make_medium, 0.3),
    ]


def _sample_matchup(
    matchups: list[MatchupSpec],
    rng: "random.Random",
) -> tuple[Agent, Agent]:
    """Sample one matchup according to weights, return two fresh agent instances."""
    weights = [w for _, _, w in matchups]
    total = sum(weights)
    threshold = rng.random() * total
    cumulative = 0.0
    for factory_a, factory_b, weight in matchups:
        cumulative += weight
        if threshold <= cumulative:
            return factory_a(), factory_b()
    # Fallback to last matchup (handles floating point edge cases)
    factory_a, factory_b, _ = matchups[-1]
    return factory_a(), factory_b()


def collect_heuristic_games(
    buf: ReplayBuffer,
    num_games: int = 200,
    matchups: list[MatchupSpec] | None = None,
) -> dict[str, int]:
    """Fill the buffer with AlphaBeta vs AlphaBeta games.

    Matchups are sampled according to the weighted matchup list. Each game
    alternates which agent plays p0/p1 for symmetric training data. Both
    agents' perspectives are recorded every game.

    Policy targets come from each agent's policy_distribution — for
    AlphaBeta this is a softmax over root move scores, not one-hot.

    Args:
        buf:      Replay buffer to push examples into.
        num_games: Number of games to play.
        matchups:  Weighted matchup specs. Defaults to easy/medium variety.
    """
    import random as random_module

    if matchups is None:
        matchups = _default_matchups()

    rng = random_module.Random()
    wins_by_p0 = 0
    wins_by_p1 = 0
    ties = 0
    all_scores: list[int] = []

    for game_num in range(num_games):
        game = Game()
        game.setup_round()

        agent_a, agent_b = _sample_matchup(matchups, rng)

        # Alternate who plays p0 for symmetric data.
        if game_num % 2 == 0:
            agents: list[Agent] = [agent_a, agent_b]
        else:
            agents = [agent_b, agent_a]

        history = _play_heuristic_game(game, agents)

        scores = _compute_game_scores(game)
        all_scores.extend(scores)

        if scores[0] > scores[1]:
            wins_by_p0 += 1
        elif scores[1] > scores[0]:
            wins_by_p1 += 1
        else:
            ties += 1

        for player_idx, spatial, flat, policy_vec in history:
            vw = win_loss_value(scores, player_idx)
            vd = score_differential_value(scores, player_idx)
            va = total_score_value(scores, player_idx)
            buf.push(spatial, flat, policy_vec, vw, vd, va)

        logger.debug(
            f"heuristic game {game_num + 1}/{num_games} -- "
            f"{type(agents[0]).__name__} vs {type(agents[1]).__name__} -- "
            f"scores {scores} -- buffer size {len(buf)}"
        )

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    logger.info(
        f"heuristic games complete -- "
        f"p0: {wins_by_p0}W / p1: {wins_by_p1}W / {ties}T -- "
        f"avg score: {avg_score:.1f} -- "
        f"{num_games} games recorded"
    )

    return {
        "wins_by_p0": wins_by_p0,
        "wins_by_p1": wins_by_p1,
        "ties": ties,
        "games_recorded": num_games,
    }


def _play_heuristic_game(
    game: "Game",
    agents: list[Agent],
) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Play a single heuristic game to completion, collecting history.

    Each history entry is (player_index, spatial, flat, policy_vec) where
    policy_vec comes from the acting agent's policy_distribution method.

    For AlphaBeta agents, policy_distribution returns uniform over all legal
    moves (inherited from Agent base class). This is soft — dozens of moves
    share probability mass — so it does not produce one-hot targets.
    """
    history: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    while not game.is_game_over():
        current_player = game.state.current_player
        spatial, flat = encode_state(game)

        agent = agents[current_player]
        # choose_move must come first — for AlphaBeta it populates the
        # internal score cache that policy_distribution reads from.
        move = agent.choose_move(game)
        policy_pairs = agent.policy_distribution(game)
        policy_vec = torch.zeros(MOVE_SPACE_SIZE)
        for m, prob in policy_pairs:
            policy_vec[encode_move(m, game)] = prob

        history.append((current_player, spatial, flat, policy_vec))
        game.make_move(move)
        game.advance()

    return history
