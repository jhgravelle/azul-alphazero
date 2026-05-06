# neural/trainer.py
"""Training utilities for AzulNet."""

from __future__ import annotations

import logging
import random
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from agents.base import Agent
from engine.game import Game
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.encoder import (
    encode_state,
    encode_move,
    MOVE_SPACE_SIZE,
    flat_policy_to_3head_targets,
)

logger = logging.getLogger(__name__)

_SCORE_DIFF_DIVISOR = 50.0
_TOTAL_SCORE_DIVISOR = 80.0
_AUX_WEIGHT_WIN = 0.3
_AUX_WEIGHT_DIFF = 1.0


# ── Value target functions ────────────────────────────────────────────────────


def win_loss_value(scores: list[int], player_index: int) -> float:
    """+1 if this player won, -1 if lost, 0 if tied."""
    own = scores[player_index]
    opp = scores[1 - player_index]
    if own > opp:
        return 1.0
    if own < opp:
        return -1.0
    return 0.0


def score_differential_value(scores: list[int], player_index: int) -> float:
    """Normalized score-differential value target for a player."""
    diff = (scores[player_index] - scores[1 - player_index]) / _SCORE_DIFF_DIVISOR
    return max(-1.0, min(1.0, diff))


def total_score_value(scores: list[int], player_index: int) -> float:
    """Normalized absolute-score value target for a player."""
    value = scores[player_index] / _TOTAL_SCORE_DIVISOR
    return max(-1.0, min(1.0, value))


# ── Loss ──────────────────────────────────────────────────────────────────────


def compute_loss(
    net: AzulNet,
    encodings: torch.Tensor,
    policies: torch.Tensor,
    values_win: torch.Tensor,
    values_diff: torch.Tensor,
    values_abs: torch.Tensor,
    policy_masks: torch.Tensor,
    value_only: bool = False,
    diff_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Combined policy + multi-head value loss.

    Args:
        policy_masks: (batch, 1) float — 1.0 = train policy, 0.0 = value-only
                      (round-boundary examples have no policy target).
        value_only: Zero out policy loss.
        diff_only:  Zero out value_win and value_abs; also zeros policy loss.

    Returns a dict with keys: total, policy, value, value_win, value_diff, value_abs.
    value_abs is diagnostic only — never included in total.
    """
    (src_logits, tile_logits, dst_logits), pred_win, pred_diff, pred_abs = net(
        encodings
    )

    if value_only or diff_only:
        policy_loss = torch.tensor(0.0, requires_grad=False)
    else:
        src_tgt, tile_tgt, dst_tgt = flat_policy_to_3head_targets(policies)
        src_loss = -(src_tgt * F.log_softmax(src_logits, dim=1)).sum(
            dim=1, keepdim=True
        )
        tile_loss = -(tile_tgt * F.log_softmax(tile_logits, dim=1)).sum(
            dim=1, keepdim=True
        )
        dst_loss = -(dst_tgt * F.log_softmax(dst_logits, dim=1)).sum(
            dim=1, keepdim=True
        )
        n_valid = policy_masks.sum().clamp(min=1.0)
        policy_loss = ((src_loss + tile_loss + dst_loss) * policy_masks).sum() / n_valid

    loss_diff = F.mse_loss(pred_diff, values_diff)
    loss_abs = F.mse_loss(pred_abs, values_abs)

    if diff_only:
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
                k: 0.0
                for k in (
                    "total",
                    "policy",
                    "value",
                    "value_win",
                    "value_diff",
                    "value_abs",
                )
            }

        encodings, policies, vw, vd, va, masks = buf.sample(self.batch_size)
        encodings = encodings.to(self.device)
        policies = policies.to(self.device)
        vw = vw.to(self.device)
        vd = vd.to(self.device)
        va = va.to(self.device)
        masks = masks.to(self.device)

        self.net.train()
        self.optimizer.zero_grad()
        loss_dict = compute_loss(
            self.net,
            encodings,
            policies,
            vw,
            vd,
            va,
            masks,
            value_only=value_only,
            diff_only=diff_only,
        )
        loss_dict["total"].backward()
        self.optimizer.step()

        return {
            k: float(v.item() if hasattr(v, "item") else v)
            for k, v in loss_dict.items()
        }


# ── Agent specs ───────────────────────────────────────────────────────────────


@dataclass
class AgentSpec:
    """Serializable description of one player in a game.

    Passed to worker processes, which reconstruct the agent from this spec.
    All fields must be picklable.

    For AlphaZero agents:
        type="alphazero", state_dict=net.state_dict(), simulations=N

    For heuristic agents:
        type="alphabeta", depths=(2,3,7), thresholds=(20,10)
        type="greedy" | "cautious" | "efficient" | "random"
    """

    type: str
    state_dict: dict | None = None
    simulations: int = 100
    depths: tuple[int, int, int] = (2, 3, 7)
    thresholds: tuple[int, int] = (20, 10)
    temperature: float = 1.0


def _build_agent(spec: AgentSpec) -> Agent:
    """Construct an agent from a spec. Runs inside worker processes."""
    from agents.alphabeta import AlphaBetaAgent
    from agents.alphazero import AlphaZeroAgent
    from agents.cautious import CautiousAgent
    from agents.efficient import EfficientAgent
    from agents.greedy import GreedyAgent
    from agents.random import RandomAgent

    if spec.type == "alphazero":
        assert spec.state_dict is not None, "AlphaZero AgentSpec requires a state_dict"
        net = AzulNet()
        net.load_state_dict(spec.state_dict)
        net.eval()
        return AlphaZeroAgent(
            net, simulations=spec.simulations, temperature=spec.temperature
        )
    if spec.type == "alphabeta":
        return AlphaBetaAgent(depths=spec.depths, thresholds=spec.thresholds)
    if spec.type == "greedy":
        return GreedyAgent()
    if spec.type == "cautious":
        return CautiousAgent()
    if spec.type == "efficient":
        return EfficientAgent()
    if spec.type == "random":
        return RandomAgent()
    raise ValueError(f"Unknown agent type: {spec.type!r}")


# ── Display helpers ───────────────────────────────────────────────────────────


def _spec_name(spec: AgentSpec) -> str:
    """Short human-readable name for an agent spec."""
    if spec.type == "alphazero":
        return f"AlphaZero(sims={spec.simulations})"
    if spec.type == "alphabeta":
        return f"AlphaBeta{spec.depths}"
    return spec.type.capitalize()


def _result_char(score_mine: int, score_theirs: int) -> str:
    """Return +, -, or * for a single game result from one player's perspective."""
    if score_mine > score_theirs:
        return "+"
    if score_mine < score_theirs:
        return "-"
    return "*"


def _format_pair_log(
    spec_0: AgentSpec,
    spec_1: AgentSpec,
    scores_a: list[int],
    scores_b: list[int],
    pair_num: int,
    total_pairs: int,
) -> str:
    """Format a mirror pair result line.

    Game A: spec_0 is p0, spec_1 is p1
    Game B: spec_1 is p0, spec_0 is p1

    Example:
        pair 3/100  RandomAgent -+  [18,32][21,15]  +- AlphaBeta(1, 2, 3)
    """
    r0_a = _result_char(scores_a[0], scores_a[1])
    r0_b = _result_char(scores_b[1], scores_b[0])
    r1_a = _result_char(scores_a[1], scores_a[0])
    r1_b = _result_char(scores_b[0], scores_b[1])
    return (
        f"pair {pair_num}/{total_pairs}  "
        f"{_spec_name(spec_0)} {r0_a}{r0_b}  "
        f"[{scores_a[0]},{scores_a[1]}][{scores_b[0]},{scores_b[1]}]  "
        f"{r1_a}{r1_b} {_spec_name(spec_1)}"
    )


# ── Game record types ─────────────────────────────────────────────────────────

# Each move: (player_idx, encoding_list, policy_list, vw, vd, va, policy_valid)
# policy_valid=False for round-boundary value-only examples (no policy target)
_MoveRecord = tuple[int, list, list, float, float, float, bool]
_GameRecord = list[_MoveRecord]


def _compute_game_scores(game: Game) -> list[int]:
    """Return earned score for each player (unclamped, includes pending and penalty)."""
    return [player.score for player in game.players]


def _play_game(
    game: Game,
    agents: list[Agent],
) -> tuple[list[tuple[int, torch.Tensor, torch.Tensor, bool]], list[int]]:
    """Play a single game to completion.

    Returns (history, scores) where history is a list of
    (player_index, encoding, policy_vec, policy_valid) tuples.
    policy_valid=False for round-boundary entries (value target only, no policy).
    """
    from agents.alphazero import AlphaZeroAgent

    # Reset any AlphaZero trees for the new game
    for agent in agents:
        if isinstance(agent, AlphaZeroAgent):
            agent.reset_tree(game)

    history: list[tuple[int, torch.Tensor, torch.Tensor, bool]] = []
    prev_round = game.round

    while True:
        if not game.legal_moves():
            break
        current_player = game.current_player_index
        encoding = encode_state(game)
        agent = agents[current_player]

        if isinstance(agent, AlphaZeroAgent):
            move, policy_pairs = agent.get_policy_targets(game)
        else:
            move = agent.choose_move(game)
            policy_pairs = agent.policy_distribution(game)

        policy_vec = torch.zeros(MOVE_SPACE_SIZE)
        for m, prob in policy_pairs:
            policy_vec[encode_move(m, game)] = prob

        history.append((current_player, encoding, policy_vec, True))

        game.make_move(move)

        # Capture round-boundary state before advance() scores and resets the round.
        # This matches the state MCTS evaluates at is_round_boundary leaf nodes:
        # empty factories, committed pattern lines, pending scores not yet settled.
        # Encoding uses next_player() to match MCTS child construction exactly.
        if game.is_round_over() and not game.is_game_over():
            boundary_game = game.clone()
            boundary_game.next_player()
            boundary_enc = encode_state(boundary_game)
            null_policy = torch.zeros(MOVE_SPACE_SIZE)
            history.append(
                (boundary_game.current_player_index, boundary_enc, null_policy, False)
            )

        game.advance()

        if game.is_game_over():
            break

        # Notify AlphaZero agents of round boundaries and moves
        if game.round != prev_round:
            for agent in agents:
                if isinstance(agent, AlphaZeroAgent):
                    agent.reset_tree(game)
            prev_round = game.round
        else:
            for agent in agents:
                if isinstance(agent, AlphaZeroAgent):
                    agent.advance(move)

    return history, _compute_game_scores(game)


def _history_to_records(
    history: list[tuple[int, torch.Tensor, torch.Tensor, bool]],
    scores: list[int],
) -> _GameRecord:
    """Convert a game history to serializable move records with value targets."""
    records: _GameRecord = []
    for player_idx, encoding, policy_vec, policy_valid in history:
        vw = win_loss_value(scores, player_idx)
        vd = score_differential_value(scores, player_idx)
        va = total_score_value(scores, player_idx)
        records.append(
            (
                player_idx,
                encoding.tolist(),
                policy_vec.tolist(),
                vw,
                vd,
                va,
                policy_valid,
            )
        )
    return records


def _push_records(buf: ReplayBuffer, records: _GameRecord) -> None:
    """Push serializable move records into the replay buffer."""
    for _player_idx, encoding_list, policy_list, vw, vd, va, policy_valid in records:
        encoding = torch.tensor(encoding_list, dtype=torch.float32)
        policy_vec = torch.tensor(policy_list, dtype=torch.float32)
        buf.push(
            encoding, policy_vec, vw, vd, va, policy_mask=1.0 if policy_valid else 0.0
        )


# ── Worker ────────────────────────────────────────────────────────────────────


def _worker_play_mirror_pair(
    spec_0: AgentSpec,
    spec_1: AgentSpec,
) -> tuple[_GameRecord, list[int], _GameRecord, list[int]]:
    """Play one mirror pair: game A then game B with sides swapped, same seed.

    Runs inside a worker process. Picks its own seed. Returns serializable
    records and scores for both games.

    Game A: spec_0 as p0, spec_1 as p1
    Game B: spec_1 as p0, spec_0 as p1 (sides swapped, same factories)
    """
    seed = random.randint(0, 2**31)
    agent_0 = _build_agent(spec_0)
    agent_1 = _build_agent(spec_1)

    game_a = Game(seed=seed)
    game_a.setup_round()
    history_a, scores_a = _play_game(game_a, [agent_0, agent_1])
    records_a = _history_to_records(history_a, scores_a)

    # Clone agents to discard any internal state from game A
    agent_0b = _build_agent(spec_0)
    agent_1b = _build_agent(spec_1)

    game_b = Game(seed=seed)
    game_b.setup_round()
    history_b, scores_b = _play_game(game_b, [agent_1b, agent_0b])
    records_b = _history_to_records(history_b, scores_b)

    return records_a, scores_a, records_b, scores_b


# ── Pool helpers ─────────────────────────────────────────────────────────────


def _worker_play_mirror_pair_tuple(
    args: tuple[int, AgentSpec, AgentSpec],
) -> tuple[int, _GameRecord, list[int], _GameRecord, list[int]]:
    """Tuple-unpacking wrapper for imap_unordered. Returns pair index alongside
    results so the main process can look up specs without passing them back.
    """
    pair_index, spec_0, spec_1 = args
    records_a, scores_a, records_b, scores_b = _worker_play_mirror_pair(spec_0, spec_1)
    return pair_index, records_a, scores_a, records_b, scores_b


def _iter_pair_results(
    sampled: list[tuple[AgentSpec, AgentSpec]],
    num_workers: int,
):
    """Yield (spec_0, spec_1, records_a, scores_a, records_b, scores_b) as each
    pair completes. Uses imap_unordered so results stream in as workers finish.
    """
    import multiprocessing as mp

    args_list = [(i, s0, s1) for i, (s0, s1) in enumerate(sampled)]

    if num_workers <= 1:
        for i, spec_0, spec_1 in args_list:
            records_a, scores_a, records_b, scores_b = _worker_play_mirror_pair(
                spec_0, spec_1
            )
            yield spec_0, spec_1, records_a, scores_a, records_b, scores_b
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            try:
                for result in pool.imap_unordered(
                    _worker_play_mirror_pair_tuple, args_list
                ):
                    pair_index, records_a, scores_a, records_b, scores_b = result
                    spec_0, spec_1 = sampled[pair_index]
                    yield spec_0, spec_1, records_a, scores_a, records_b, scores_b
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                raise


# ── Unified parallel collection ───────────────────────────────────────────────


def collect_parallel(
    buf: ReplayBuffer,
    spec_0: AgentSpec,
    spec_1: AgentSpec,
    num_pairs: int,
    num_workers: int = 1,
) -> dict[str, int]:
    """Play mirror pairs and push examples into buf.

    Each pair plays game A (spec_0 as p0) and game B (sides swapped) with
    identical factory sequences. Both perspectives recorded from both games.

    Falls back to sequential if num_workers <= 1.

    Returns stats: wins_0, wins_1, ties, games_recorded.
    """
    wins_0 = 0
    wins_1 = 0
    ties = 0

    for pair_num, (
        spec_0,
        spec_1,
        records_a,
        scores_a,
        records_b,
        scores_b,
    ) in enumerate(
        _iter_pair_results([(spec_0, spec_1)] * num_pairs, num_workers), start=1
    ):
        _push_records(buf, records_a)
        _push_records(buf, records_b)

        if scores_a[0] > scores_a[1]:
            wins_0 += 1
        elif scores_a[1] > scores_a[0]:
            wins_1 += 1
        else:
            ties += 1

        if scores_b[1] > scores_b[0]:
            wins_0 += 1
        elif scores_b[0] > scores_b[1]:
            wins_1 += 1
        else:
            ties += 1

        logger.info(
            _format_pair_log(spec_0, spec_1, scores_a, scores_b, pair_num, num_pairs)
        )

    games_recorded = num_pairs * 2
    logger.info(
        f"collect_parallel complete -- "
        f"spec_0: {wins_0}W / spec_1: {wins_1}W / {ties}T -- "
        f"{games_recorded} games recorded"
    )
    return {
        "wins_0": wins_0,
        "wins_1": wins_1,
        "ties": ties,
        "games_recorded": games_recorded,
    }


# ── Pretrain matchups ─────────────────────────────────────────────────────────


def _pretrain_matchups() -> list[tuple[AgentSpec, AgentSpec, float]]:
    """Weak heuristics vs easy AlphaBeta for fast buffer initialization.

    Returns a list of (spec_0, spec_1, weight) triples.
    """
    easy = AgentSpec(type="alphabeta", depths=(1, 2, 3), thresholds=(20, 10))
    return [
        (AgentSpec(type="random"), easy, 0.25),
        (AgentSpec(type="efficient"), easy, 0.25),
        (AgentSpec(type="cautious"), easy, 0.25),
        (AgentSpec(type="greedy"), easy, 0.25),
    ]


def _default_matchups() -> list[tuple[AgentSpec, AgentSpec, float]]:
    """Default weighted matchup list for heuristic data collection.

    All matchups pair heuristic agents against AlphaBeta easy.
    AlphaBeta easy runs fast (~4ms/move) allowing high game volume.

    random    vs easy: 10% — extreme loss signal
    efficient vs easy: 20% — weak vs strong
    cautious  vs easy: 30% — moderate loss signal
    greedy    vs easy: 40% — near-peer, clean policy targets
    """
    easy = AgentSpec(type="alphabeta", depths=(1, 2, 3), thresholds=(20, 10))
    return [
        (AgentSpec(type="random"), easy, 0.10),
        (AgentSpec(type="efficient"), easy, 0.20),
        (AgentSpec(type="cautious"), easy, 0.30),
        (AgentSpec(type="greedy"), easy, 0.40),
    ]


def _sample_matchup(
    matchups: list[tuple[AgentSpec, AgentSpec, float]],
) -> tuple[AgentSpec, AgentSpec]:
    """Sample one matchup according to weights."""
    weights = [w for _, _, w in matchups]
    total = sum(weights)
    threshold = random.random() * total
    cumulative = 0.0
    for spec_0, spec_1, weight in matchups:
        cumulative += weight
        if threshold <= cumulative:
            return spec_0, spec_1
    spec_0, spec_1, _ = matchups[-1]
    return spec_0, spec_1


def collect_heuristic_parallel(
    buf: ReplayBuffer,
    num_pairs: int,
    matchups: list[tuple[AgentSpec, AgentSpec, float]] | None = None,
    num_workers: int = 1,
) -> dict[str, int]:
    """Collect heuristic mirror pairs sampled from a weighted matchup list.

    Each pair samples independently, so a run of 100 pairs with 4 matchups
    at equal weight produces ~25 pairs of each type.
    """
    if matchups is None:
        matchups = _default_matchups()

    wins_0 = 0
    wins_1 = 0
    ties = 0
    games_recorded = 0

    # Sample matchups first so workers get concrete specs
    sampled = [_sample_matchup(matchups) for _ in range(num_pairs)]

    for pair_num, (
        spec_0,
        spec_1,
        records_a,
        scores_a,
        records_b,
        scores_b,
    ) in enumerate(_iter_pair_results(sampled, num_workers), start=1):
        _push_records(buf, records_a)
        _push_records(buf, records_b)
        games_recorded += 2

        if scores_a[0] > scores_a[1]:
            wins_0 += 1
        elif scores_a[1] > scores_a[0]:
            wins_1 += 1
        else:
            ties += 1

        if scores_b[1] > scores_b[0]:
            wins_0 += 1
        elif scores_b[0] > scores_b[1]:
            wins_1 += 1
        else:
            ties += 1

        logger.info(
            _format_pair_log(spec_0, spec_1, scores_a, scores_b, pair_num, num_pairs)
        )

    logger.info(
        f"heuristic collection complete -- "
        f"wins_0: {wins_0} / wins_1: {wins_1} / ties: {ties} -- "
        f"{games_recorded} games recorded"
    )
    return {
        "wins_0": wins_0,
        "wins_1": wins_1,
        "ties": ties,
        "games_recorded": games_recorded,
    }


# ── Eval ──────────────────────────────────────────────────────────────────────


def evaluate_parallel(
    new_net: AzulNet,
    old_net: AzulNet,
    num_pairs: int,
    simulations: int,
    buf: ReplayBuffer | None,
    num_workers: int = 1,
) -> float:
    """Evaluate new_net vs old_net using parallel mirror pairs.

    Each pair plays game A (new as p0) and game B (new as p1) with the same
    seed. All pairs run to completion — no early exit.

    If buf is provided, all game history is pushed into the replay buffer.

    Returns new net win rate over all games played.
    """
    new_spec = AgentSpec(
        type="alphazero",
        state_dict={k: v.cpu() for k, v in new_net.state_dict().items()},
        simulations=simulations,
        temperature=0.0,
    )
    old_spec = AgentSpec(
        type="alphazero",
        state_dict={k: v.cpu() for k, v in old_net.state_dict().items()},
        simulations=simulations,
        temperature=0.0,
    )

    new_wins = 0.0
    games_played = 0

    for pair_num, (_, _, records_a, scores_a, records_b, scores_b) in enumerate(
        _iter_pair_results([(new_spec, old_spec)] * num_pairs, num_workers), start=1
    ):
        games_played += 2

        if scores_a[0] > scores_a[1]:
            new_wins += 1.0
        elif scores_a[0] == scores_a[1]:
            new_wins += 0.5

        if scores_b[1] > scores_b[0]:
            new_wins += 1.0
        elif scores_b[1] == scores_b[0]:
            new_wins += 0.5

        logger.info(
            _format_pair_log(
                new_spec, old_spec, scores_a, scores_b, pair_num, num_pairs
            )
            + f"  ({new_wins / games_played:.0%})"
        )

        if buf is not None:
            _push_records(buf, records_a)
            _push_records(buf, records_b)

    win_rate = new_wins / games_played if games_played else 0.0
    logger.info(f"eval complete -- new net win rate: {win_rate * 100:.1f}%")
    return win_rate
