# neural/trainer.py
"""Training utilities for AzulNet."""

from __future__ import annotations

import logging
import random
import torch
import torch.nn.functional as F
from typing import Callable

from agents.alphazero import AlphaZeroAgent
from agents.base import Agent
from engine.game import Game, FLOOR
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.encoder import (
    encode_state,
    encode_move,
    MOVE_SPACE_SIZE,
    flat_policy_to_3head_targets,
)

logger = logging.getLogger(__name__)

_SCORE_DIFF_DIVISOR = 50.0  # matches encoder SCORE_DELTA_DIVISOR
_TOTAL_SCORE_DIVISOR = 80.0
_AUX_WEIGHT_WIN = 0.3
_AUX_WEIGHT_DIFF = 1.0


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
    encodings: torch.Tensor,  # (B, FLAT_SIZE) = (B, 123)
    policies: torch.Tensor,  # (B, MOVE_SPACE_SIZE)
    values_win: torch.Tensor,  # (B, 1)
    values_diff: torch.Tensor,  # (B, 1)
    values_abs: torch.Tensor,  # (B, 1)
    value_only: bool = False,
    diff_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Combined policy + multi-head value loss.

    Args:
        value_only: Zero out policy loss — trunk learns from value signal only.
        diff_only:  Zero out value_win and value_abs — trunk learns from score
                    differential only. Use during Phase 1 to give the value head
                    a dense continuous signal before win/loss targets are meaningful.
                    Implies value_only=True (policy is also zeroed).

    Returns a dict with:
        total:      combined loss
        policy:     cross-entropy on MCTS visit distribution (0 if value_only)
        value:      combined value loss
        value_win:  MSE on win/loss target (0 if diff_only)
        value_diff: MSE on score-differential target (always active)
        value_abs:  MSE on absolute-score target — diagnostic only, not in total
    """
    (src_logits, tile_logits, dst_logits), pred_win, pred_diff, pred_abs = net(
        encodings
    )

    if value_only or diff_only:
        policy_loss = torch.tensor(0.0, requires_grad=False)
    else:
        src_tgt, tile_tgt, dst_tgt = flat_policy_to_3head_targets(policies)
        policy_loss = (
            -(src_tgt * F.log_softmax(src_logits, dim=1)).sum(dim=1).mean()
            + -(tile_tgt * F.log_softmax(tile_logits, dim=1)).sum(dim=1).mean()
            + -(dst_tgt * F.log_softmax(dst_logits, dim=1)).sum(dim=1).mean()
        )

    loss_diff = F.mse_loss(pred_diff, values_diff)
    # value_abs is always computed for logging but excluded from training loss
    loss_abs = F.mse_loss(pred_abs, values_abs)

    if diff_only:
        # Only score differential — gives trunk a dense continuous training signal
        # without the noise of win/loss targets on early-training data.
        loss_win = torch.tensor(0.0, requires_grad=False)
        combined_value = loss_diff
    else:
        loss_win = F.mse_loss(pred_win, values_win)
        combined_value = _AUX_WEIGHT_WIN * loss_win + _AUX_WEIGHT_DIFF * loss_diff

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
        self,
        buf: ReplayBuffer,
        value_only: bool = False,
        diff_only: bool = False,
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
        encodings, policies, vw, vd, va = buf.sample(self.batch_size)
        encodings = encodings.to(self.device)
        policies = policies.to(self.device)
        vw = vw.to(self.device)
        vd = vd.to(self.device)
        va = va.to(self.device)
        self.net.train()
        self.optimizer.zero_grad()
        loss_dict = compute_loss(
            self.net,
            encodings,
            policies,
            vw,
            vd,
            va,
            value_only=value_only,
            diff_only=diff_only,
        )
        loss_dict["total"].backward()
        self.optimizer.step()
        return {
            k: float(v.item() if hasattr(v, "item") else v)
            for k, v in loss_dict.items()
        }


# ── Self-play data collection ─────────────────────────────────────────────────


def _compute_game_scores(game: Game) -> list[int]:
    """Return earned score for each player.

    Uses player.earned (score + pending + penalty + bonus) so floor
    penalties below zero carry meaningful signal even when score is clamped.
    """
    return [player.earned for player in game.players]


def collect_self_play(
    buf: ReplayBuffer,
    net: AzulNet,
    num_games: int = 50,
    simulations: int = 100,
    temperature: float = 1.0,
    opponent: Agent | None = None,
    _device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Play num_games games and push training examples into buf."""
    from engine.game import Game

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

        history: list[tuple[int, torch.Tensor, torch.Tensor]] = []

        while True:
            if not game.legal_moves():
                break
            current_player = game.current_player_index
            is_az_turn = az_player is None or current_player == az_player

            encoding = encode_state(game)

            if is_az_turn:
                move, policy_pairs = az_agent.get_policy_targets(game)
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
                policy_pairs = agents[current_player].policy_distribution(game)
                policy_vec = torch.zeros(MOVE_SPACE_SIZE)
                for m, prob in policy_pairs:
                    policy_vec[encode_move(m, game)] = prob

            history.append((current_player, encoding, policy_vec))

            prev_round = game.round
            game.make_move(move)
            game.advance()

            if game.is_game_over():
                break

            if game.round != prev_round:
                az_agent.reset_tree(game)
            else:
                az_agent.advance(move)

        scores = _compute_game_scores(game)

        for player_idx, encoding, policy_vec in history:
            vw = win_loss_value(scores, player_idx)
            vd = score_differential_value(scores, player_idx)
            va = total_score_value(scores, player_idx)
            buf.push(encoding, policy_vec, vw, vd, va)

        az_score = scores[az_player] if az_player is not None else max(scores)
        az_scores.append(az_score)

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


def _pretrain_matchups() -> list[MatchupSpec]:
    """Weak heuristics vs easy AlphaBeta for fast buffer initialization."""
    from agents.alphabeta import AlphaBetaAgent
    from agents.random import RandomAgent
    from agents.efficient import EfficientAgent
    from agents.cautious import CautiousAgent
    from agents.greedy import GreedyAgent

    def make_random() -> RandomAgent:
        return RandomAgent()

    def make_efficient() -> EfficientAgent:
        return EfficientAgent()

    def make_cautious() -> CautiousAgent:
        return CautiousAgent()

    def make_greedy() -> GreedyAgent:
        return GreedyAgent()

    def make_easy() -> AlphaBetaAgent:
        return AlphaBetaAgent(depths=(1, 2, 3), thresholds=(20, 10))

    return [
        (make_random, make_easy, 0.25),
        (make_efficient, make_easy, 0.25),
        (make_cautious, make_easy, 0.25),
        (make_greedy, make_easy, 0.25),
    ]


def _default_matchups() -> list[MatchupSpec]:
    """Default weighted matchup list for heuristic data collection.

    All matchups pair a variety of skill levels against AlphaBeta easy,
    so every game has one reference player without the speed cost of medium.
    AlphaBeta easy runs ~8x faster than medium (~4ms vs ~35ms per move),
    allowing significantly more games per iteration.

    Random vs easy:    10% -- extreme loss signal, fast games
    Efficient vs easy: 10% -- weak vs strong, passive play exposed
    Cautious vs easy:  15% -- moderate loss signal, floor avoidance
    Greedy vs easy:    20% -- near-peer, clean policy targets
    Easy vs easy:      45% -- symmetric, consistent quality
    """
    from agents.alphabeta import AlphaBetaAgent
    from agents.random import RandomAgent
    from agents.efficient import EfficientAgent
    from agents.cautious import CautiousAgent
    from agents.greedy import GreedyAgent

    def make_random() -> RandomAgent:
        return RandomAgent()

    def make_efficient() -> EfficientAgent:
        return EfficientAgent()

    def make_cautious() -> CautiousAgent:
        return CautiousAgent()

    def make_greedy() -> GreedyAgent:
        return GreedyAgent()

    def make_easy() -> AlphaBetaAgent:
        return AlphaBetaAgent(depths=(1, 2, 3), thresholds=(20, 10))

    def make_medium() -> AlphaBetaAgent:
        return AlphaBetaAgent(depths=(2, 3, 7), thresholds=(20, 10))

    return [
        (make_random, make_easy, 0.00),
        (make_efficient, make_easy, 0.00),
        (make_cautious, make_easy, 0.00),
        (make_greedy, make_easy, 0.00),
        (make_easy, make_easy, 0.00),
        (make_easy, make_medium, 1.00),
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


def _clone_agent(agent: Agent) -> Agent:
    """Return a fresh instance of the same agent type with identical config.

    Needed before mirror games — agents may carry internal state (e.g.
    AlphaBeta score cache) from a previous game. Always construct fresh.
    """
    from agents.alphabeta import AlphaBetaAgent
    from agents.random import RandomAgent
    from agents.efficient import EfficientAgent
    from agents.cautious import CautiousAgent
    from agents.greedy import GreedyAgent

    if isinstance(agent, AlphaBetaAgent):
        return AlphaBetaAgent(depths=agent.depths, thresholds=agent.thresholds)
    if isinstance(agent, RandomAgent):
        return RandomAgent()
    if isinstance(agent, EfficientAgent):
        return EfficientAgent()
    if isinstance(agent, CautiousAgent):
        return CautiousAgent()
    if isinstance(agent, GreedyAgent):
        return GreedyAgent()
    raise ValueError(f"Cannot clone agent type: {type(agent)}")


def collect_mirror_heuristic_games(
    buf: ReplayBuffer,
    num_pairs: int = 100,
    matchups: list[MatchupSpec] | None = None,
) -> dict[str, int]:
    """Play pairs of mirror games with identical factory sequences.

    For each pair, two agents play both games with sides swapped. Both
    games use Game(seed=N) so the bag shuffle — and therefore all factory
    draws across all rounds — is identical. Since a normal 2-player game
    draws exactly 100 tiles across 5 rounds and the bag holds 100 tiles,
    no refill occurs and the seed fully determines all factories.

    If the stronger agent wins from both sides, the factory configuration
    has zero net correlation with outcome in this pair. Over many pairs
    this forces the value head to ignore factory fingerprints.

    Policy targets come from each agent's policy_distribution.
    Both players' perspectives are recorded for every game.
    """
    if matchups is None:
        matchups = _default_matchups()

    rng = random.Random()
    wins_by_p0 = 0
    wins_by_p1 = 0
    ties = 0

    for pair_num in range(num_pairs):
        agent_a, agent_b = _sample_matchup(matchups, rng)
        game_seed = rng.randint(0, 2**31)

        # Game 1: agent_a as p0, agent_b as p1
        game_1 = Game(seed=game_seed)
        game_1.setup_round()
        history_1 = _play_heuristic_game(game_1, [agent_a, agent_b])
        scores_1 = _compute_game_scores(game_1)

        # Game 2: sides swapped, identical factory sequence via same seed.
        # Clone agents to discard any internal state from game 1.
        game_2 = Game(seed=game_seed)
        game_2.setup_round()
        history_2 = _play_heuristic_game(
            game_2, [_clone_agent(agent_b), _clone_agent(agent_a)]
        )
        scores_2 = _compute_game_scores(game_2)

        for scores, history in [(scores_1, history_1), (scores_2, history_2)]:
            if scores[0] > scores[1]:
                wins_by_p0 += 1
            elif scores[1] > scores[0]:
                wins_by_p1 += 1
            else:
                ties += 1
            for player_idx, encoding, policy_vec in history:
                vw = win_loss_value(scores, player_idx)
                vd = score_differential_value(scores, player_idx)
                va = total_score_value(scores, player_idx)
                buf.push(encoding, policy_vec, vw, vd, va)

        logger.debug(
            f"mirror pair {pair_num + 1}/{num_pairs} -- seed {game_seed} -- "
            f"game1 scores {scores_1} -- game2 scores {scores_2}"
        )

    games_recorded = num_pairs * 2
    logger.info(
        f"mirror games complete -- "
        f"p0: {wins_by_p0}W / p1: {wins_by_p1}W / {ties}T -- "
        f"{games_recorded} games recorded"
    )
    return {
        "wins_by_p0": wins_by_p0,
        "wins_by_p1": wins_by_p1,
        "ties": ties,
        "games_recorded": games_recorded,
    }


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

        for player_idx, encoding, policy_vec in history:
            vw = win_loss_value(scores, player_idx)
            vd = score_differential_value(scores, player_idx)
            va = total_score_value(scores, player_idx)
            buf.push(encoding, policy_vec, vw, vd, va)

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
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Play a single heuristic game to completion, collecting history.

    Each history entry is (player_index, encoding, policy_vec) where
    policy_vec comes from the acting agent's policy_distribution method.

    For AlphaBeta agents, policy_distribution returns a softmax over root
    move scores. choose_move must be called first to populate the cache.
    """
    history: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    while True:
        if not game.legal_moves():
            break
        current_player = game.current_player_index
        encoding = encode_state(game)

        agent = agents[current_player]
        move = agent.choose_move(game)
        policy_pairs = agent.policy_distribution(game)
        policy_vec = torch.zeros(MOVE_SPACE_SIZE)
        for m, prob in policy_pairs:
            policy_vec[encode_move(m, game)] = prob

        history.append((current_player, encoding, policy_vec))
        game.make_move(move)
        game.advance()

        if game.is_game_over():
            break

    return history


# ── Parallel heuristic game collection ───────────────────────────────────────

# A game record is a list of move records.
# Each move record: (player_idx, encoding_list, policy_list, vw, vd, va)
# All tensors are stored as plain Python lists to survive multiprocessing pickle.
_MoveRecord = tuple[int, list, list, float, float, float]
_GameRecord = list[_MoveRecord]


def _worker_play_games(
    num_games: int,
    worker_seed: int,
    matchup_specs: list[tuple[str, str, float]],
) -> tuple[list[_GameRecord], dict[str, int]]:
    """Worker function: play num_games heuristic games, return serializable results.

    Runs in a subprocess — no shared state with the main process.
    matchup_specs uses agent class names (strings) rather than callables
    because callables don't pickle reliably across processes.

    Returns (game_records, stats_dict).
    """
    import random as random_module
    from agents.alphabeta import AlphaBetaAgent
    from agents.random import RandomAgent
    from agents.efficient import EfficientAgent
    from agents.cautious import CautiousAgent
    from agents.greedy import GreedyAgent

    def _make_agent(name: str) -> Agent:
        if name == "random":
            return RandomAgent()
        if name == "efficient":
            return EfficientAgent()
        if name == "cautious":
            return CautiousAgent()
        if name == "greedy":
            return GreedyAgent()
        if name == "easy":
            return AlphaBetaAgent(depths=(2, 3, 7), thresholds=(20, 10))
        if name == "medium":
            return AlphaBetaAgent(depths=(3, 5, 7), thresholds=(20, 10))
        raise ValueError(f"Unknown agent name: {name}")

    rng = random_module.Random(worker_seed)
    weights = [w for _, _, w in matchup_specs]
    total_weight = sum(weights)

    wins_by_p0 = 0
    wins_by_p1 = 0
    ties = 0
    game_records: list[_GameRecord] = []

    for game_num in range(num_games):
        # Sample matchup by weight
        threshold = rng.random() * total_weight
        cumulative = 0.0
        name_a, name_b = matchup_specs[-1][0], matchup_specs[-1][1]
        for a_name, b_name, weight in matchup_specs:
            cumulative += weight
            if threshold <= cumulative:
                name_a, name_b = a_name, b_name
                break

        agent_a = _make_agent(name_a)
        agent_b = _make_agent(name_b)

        if game_num % 2 == 0:
            agents: list[Agent] = [agent_a, agent_b]
        else:
            agents = [agent_b, agent_a]

        game = Game()
        game.setup_round()
        history = _play_heuristic_game(game, agents)
        scores = _compute_game_scores(game)

        if scores[0] > scores[1]:
            wins_by_p0 += 1
        elif scores[1] > scores[0]:
            wins_by_p1 += 1
        else:
            ties += 1

        game_record: _GameRecord = []
        for player_idx, encoding, policy_vec in history:
            vw = win_loss_value(scores, player_idx)
            vd = score_differential_value(scores, player_idx)
            va = total_score_value(scores, player_idx)
            game_record.append(
                (
                    player_idx,
                    encoding.tolist(),
                    policy_vec.tolist(),
                    vw,
                    vd,
                    va,
                )
            )
        game_records.append(game_record)

    stats = {
        "wins_by_p0": wins_by_p0,
        "wins_by_p1": wins_by_p1,
        "ties": ties,
        "games_recorded": num_games,
    }
    return game_records, stats


def _matchups_to_specs(
    matchups: list[MatchupSpec],
) -> list[tuple[str, str, float]]:
    """Convert callable matchup specs to serializable (name, name, weight) tuples.

    The worker process reconstructs agents by name since callables don't
    pickle reliably across processes.
    """
    from agents.alphabeta import AlphaBetaAgent
    from agents.random import RandomAgent
    from agents.efficient import EfficientAgent
    from agents.cautious import CautiousAgent
    from agents.greedy import GreedyAgent

    _type_to_name = {
        RandomAgent: "random",
        EfficientAgent: "efficient",
        CautiousAgent: "cautious",
        GreedyAgent: "greedy",
    }

    specs = []
    for factory_a, factory_b, weight in matchups:
        agent_a = factory_a()
        agent_b = factory_b()
        if isinstance(agent_a, AlphaBetaAgent):
            name_a = "easy" if agent_a.depths == (2, 3, 7) else "medium"
        else:
            name_a = _type_to_name[type(agent_a)]
        if isinstance(agent_b, AlphaBetaAgent):
            name_b = "easy" if agent_b.depths == (2, 3, 7) else "medium"
        else:
            name_b = _type_to_name[type(agent_b)]
        specs.append((name_a, name_b, weight))
    return specs


def collect_heuristic_games_parallel(
    buf: ReplayBuffer,
    num_games: int = 200,
    matchups: list[MatchupSpec] | None = None,
    num_workers: int = 4,
) -> dict[str, int]:
    """Parallel version of collect_heuristic_games using multiprocessing.

    Splits num_games across num_workers subprocesses. Each worker plays
    its share of games independently and returns serializable results.
    The main process collects results and pushes tensors into the buffer.

    Falls back to sequential collection if num_workers <= 1.
    """
    import multiprocessing as mp
    import random

    if num_workers <= 1:
        return collect_heuristic_games(buf, num_games=num_games, matchups=matchups)

    if matchups is None:
        matchups = _default_matchups()

    specs = _matchups_to_specs(matchups)

    # Distribute games across workers as evenly as possible
    base = num_games // num_workers
    remainder = num_games % num_workers
    game_counts = [base + (1 if i < remainder else 0) for i in range(num_workers)]
    seeds = [random.randint(0, 2**31) for _ in range(num_workers)]

    args_list = [(count, seed, specs) for count, seed in zip(game_counts, seeds)]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.starmap(_worker_play_games, args_list)

    # Aggregate stats and push all examples into the buffer
    total_stats: dict[str, int] = {
        "wins_by_p0": 0,
        "wins_by_p1": 0,
        "ties": 0,
        "games_recorded": 0,
    }
    all_scores: list[float] = []

    for game_records, stats in results:
        for key in total_stats:
            total_stats[key] += stats[key]
        for game_record in game_records:
            for _player_idx, encoding_l, policy_l, vw, vd, va in game_record:
                encoding = torch.tensor(encoding_l, dtype=torch.float32)
                policy_vec = torch.tensor(policy_l, dtype=torch.float32)
                buf.push(encoding, policy_vec, vw, vd, va)
            if game_record:
                all_scores.append(vw)  # last move's vw as a proxy

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    logger.info(
        f"heuristic games complete (parallel, {num_workers} workers) -- "
        f"p0: {total_stats['wins_by_p0']}W / "
        f"p1: {total_stats['wins_by_p1']}W / {total_stats['ties']}T -- "
        f"avg score proxy: {avg_score:.2f} -- "
        f"{total_stats['games_recorded']} games recorded"
    )

    return total_stats
