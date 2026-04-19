#!/usr/bin/env python
# scripts/train.py

"""AlphaZero self-play training loop for Azul.

Usage:
    python -m scripts.train                         # all defaults
    python -m scripts.train --iterations 20 --games-per-iter 50
    python -m scripts.train --load checkpoints/gen_005.pt
    python -m scripts.train --help
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import torch

from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import (
    Trainer,
    collect_self_play,
    collect_heuristic_games,
    total_score_value,
    win_loss_value,
)
from agents.greedy import GreedyAgent
from agents.random import RandomAgent
from engine.game import Game

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
_MAX_MOVES = 100


# ── Logging setup ─────────────────────────────────────────────────────────────


def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(fh)

    # Silence engine-level noise — board/factory/bag construction and bag
    # refills are not useful during training runs.
    logging.getLogger("engine.game_state").setLevel(logging.WARNING)
    logging.getLogger("engine.game").setLevel(logging.WARNING)

    return log_path


logger = logging.getLogger(__name__)


# ── Iteration result ──────────────────────────────────────────────────────────


@dataclass
class IterResult:
    iteration: int
    mode: str  # "warmup" or "self-play"
    avg_loss: float
    win_rate: float
    promoted: bool  # did this iteration produce a new generation?
    generation: int  # generation number if promoted, else 0
    az_avg: float  # rolling avg AZ score
    elapsed: float  # seconds


# ── Loss accumulator helpers ──────────────────────────────────────────────────

_LOSS_KEYS = ("total", "policy", "value", "value_win", "value_diff", "value_abs")


def _init_loss_accumulator() -> dict[str, float]:
    return {k: 0.0 for k in _LOSS_KEYS}


def _accumulate_losses(accum: dict[str, float], step_losses: dict[str, float]) -> None:
    for k in _LOSS_KEYS:
        accum[k] += step_losses.get(k, 0.0)


def _format_loss_line(accum: dict[str, float], n_steps: int) -> str:
    """Format the per-head loss breakdown for logging."""
    avg = {k: accum[k] / n_steps for k in _LOSS_KEYS}
    return (
        f"avg loss: {avg['total']:.4f} | "
        f"policy: {avg['policy']:.4f} | "
        f"value: {avg['value']:.4f} "
        f"(win {avg['value_win']:.4f}, "
        f"diff {avg['value_diff']:.4f}, "
        f"abs {avg['value_abs']:.4f})"
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(net: AzulNet, generation: int) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"gen_{generation:04d}.pt"
    torch.save(net.state_dict(), path)
    logger.info("saved checkpoint -> %s", path)
    return path


def load_checkpoint(net: AzulNet, path: str) -> None:
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    logger.info("loaded checkpoint <- %s", path)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    new_net: AzulNet,
    old_net: AzulNet,
    num_games: int = 40,
    simulations: int = 25,
    win_threshold: float = 0.52,
    buf: ReplayBuffer | None = None,
    record: bool = False,
    iteration: int = 0,
    generation: int = 0,
) -> float:
    from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE
    from neural.trainer import score_differential_value
    from engine.game_recorder import GameRecorder
    from neural.model import AzulNet as _AzulNet
    from agents.alphazero import AlphaZeroAgent

    # MCTS inference is faster on CPU — use CPU copies of nets for tree search.
    def _to_cpu(n: AzulNet) -> AzulNet:
        if next(n.parameters()).device.type == "cpu":
            return n
        n_cpu = _AzulNet()
        n_cpu.load_state_dict(n.state_dict())
        return n_cpu

    new_net_cpu = _to_cpu(new_net)
    old_net_cpu = _to_cpu(old_net)

    wins_needed = math.ceil(num_games * win_threshold)
    losses_allowed = num_games - wins_needed

    new_wins = 0.0
    losses = 0
    games_played = 0

    for i in range(num_games):
        game = Game()
        game.setup_round()
        new_is_p0 = i % 2 == 0

        # Two agents, each with their own tree, using their own net.
        nets_cpu = (
            [new_net_cpu, old_net_cpu] if new_is_p0 else [old_net_cpu, new_net_cpu]
        )
        agents = [
            AlphaZeroAgent(nets_cpu[0], simulations=simulations, temperature=1.0),
            AlphaZeroAgent(nets_cpu[1], simulations=simulations, temperature=1.0),
        ]
        for agent in agents:
            agent.reset_tree(game)

        moves = 0
        history: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        prev_round = game.state.round

        # Set up recorder for first game only
        recorder: GameRecorder | None = None
        if record and i == 0:
            recorder = GameRecorder(
                player_names=[
                    f"candidate (iter {iteration})",
                    f"best (gen {generation:04d})",
                ],
                player_types=["alphazero", "alphazero"],
            )
            recorder.start_round(game)
        last_round = game.state.round

        while not game.is_game_over() and moves < _MAX_MOVES:
            if not game.legal_moves():
                break
            current = game.state.current_player

            if buf is not None:
                spatial, flat = encode_state(game)
                move, policy_pairs = agents[current].get_policy_targets(game)
                policy_vec = torch.zeros(MOVE_SPACE_SIZE)
                for m, prob in policy_pairs:
                    policy_vec[encode_move(m, game)] = prob
                history.append((current, spatial, flat, policy_vec))
            else:
                move = agents[current].choose_move(game)

            if recorder is not None:
                recorder.record_move(move, player_index=current)

            prev_round = game.state.round
            game.make_move(move)
            game.advance_round_if_needed()

            if game.state.round != prev_round:
                for agent in agents:
                    agent.reset_tree(game)
            elif not game.is_game_over():
                for agent in agents:
                    agent.advance(move)

            if (
                recorder is not None
                and not game.is_game_over()
                and game.state.round != last_round
            ):
                last_round = game.state.round
                recorder.start_round(game)

            moves += 1

        if moves >= _MAX_MOVES:
            logger.warning(
                f"eval game {i} hit move cap ({_MAX_MOVES}) -- scoring as tie"
            )

        if recorder is not None:
            recorder.finalize(game)
            eval_dir = Path("recordings/eval")
            eval_dir.mkdir(parents=True, exist_ok=True)
            filename = f"eval_iter_{iteration:03d}_vs_gen_{generation:04d}.json"
            recorder.save(eval_dir / filename)
            logger.info(f"saved eval recording -> {filename}")

        if buf is not None:
            scores = [p.score - p.clamped_points for p in game.state.players]
            for player_idx, spatial, flat, policy_vec in history:
                vw = win_loss_value(scores, player_idx)
                vd = score_differential_value(scores, player_idx)
                va = total_score_value(scores, player_idx)
                buf.push(spatial, flat, policy_vec, vw, vd, va)
        else:
            scores = [p.score for p in game.state.players]

        games_played += 1

        if scores[0] == scores[1]:
            result = "tie"
            new_wins += 0.5
        elif (scores[0] > scores[1]) == new_is_p0:
            result = "new ✓"
            new_wins += 1.0
        else:
            result = "old ✗"
            losses += 1

        logger.info(
            f"  eval game {i + 1}/{num_games} -- scores {scores} -- "
            f"{result} (new win rate so far: {new_wins / games_played:.0%})"
        )

        if new_wins >= wins_needed:
            logger.info(
                f"  eval early pass after {games_played}/{num_games} games -- "
                f"{new_wins:.0f} wins >= {wins_needed} needed"
            )
            break
        if losses > losses_allowed:
            logger.info(
                f"  eval early fail after {games_played}/{num_games} games -- "
                f"{losses} losses > {losses_allowed} allowed"
            )
            break

    return new_wins / num_games


def evaluate_vs_random(
    net: AzulNet,
    num_games: int = 20,
    simulations: int = 25,
) -> float:
    from neural.search_tree import SearchTree, make_policy_value_fn

    wins = 0.0
    for i in range(num_games):
        game = Game()
        game.setup_round()
        az_is_p0 = i % 2 == 0
        rng_agent = RandomAgent()
        moves = 0
        while not game.is_game_over() and moves < _MAX_MOVES:
            if not game.legal_moves():
                break
            current = game.state.current_player
            is_az = (current == 0) == az_is_p0
            if is_az:
                tree = SearchTree(
                    policy_value_fn=make_policy_value_fn(net, DEVICE),
                    simulations=simulations,
                    temperature=0.0,
                )
                move = tree.choose_move(game)
            else:
                move = rng_agent.choose_move(game)
            game.make_move(move)
            game.advance_round_if_needed()
            moves += 1
        scores = [p.score for p in game.state.players]
        if scores[0] == scores[1]:
            wins += 0.5
        elif (scores[0] > scores[1]) == az_is_p0:
            wins += 1.0
    return wins / num_games


# ── Summary helpers ───────────────────────────────────────────────────────────


def _summary_line(r: IterResult) -> str:
    promoted = f"✓ gen{r.generation:04d}" if r.promoted else "✗      "
    return (
        f"iter {r.iteration:3d} | loss {r.avg_loss:.4f} | "
        f"win {r.win_rate * 100:5.1f}% {promoted} | "
        f"az-avg {r.az_avg:5.1f} | {r.mode} | {r.elapsed:.0f}s"
    )


def print_summary(results: list[IterResult], generation: int) -> None:
    sep = "-" * 72
    logger.info(sep)
    logger.info("Training Summary")
    logger.info(sep)
    for r in results:
        logger.info(_summary_line(r))
    logger.info(sep)
    logger.info(
        "total generations: %d | best checkpoint: checkpoints/gen_%04d.pt",
        generation,
        generation,
    )
    logger.info(sep)


# ── Main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Azul AlphaZero training loop")
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="number of generate->train->eval cycles (default 30)",
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=25,
        help="self-play games per iteration (default 25)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=500,
        help="gradient steps per iteration (default 500)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move during self-play (default 100)",
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=25,
        help="MCTS simulations per move during evaluation (default 25)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=40,
        help="games for new-vs-old evaluation (default 40)",
    )
    parser.add_argument(
        "--win-threshold",
        type=float,
        default=0.52,
        help="new model win rate required to replace old (default 0.52)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="replay buffer capacity (default 50000)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="training batch size (default 256)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Adam learning rate (default 0.001)"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="path to a checkpoint to resume from"
    )
    parser.add_argument(
        "--pretrain-games",
        type=int,
        default=0,
        help="heuristic games to fill buffer before self-play (default 0)",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=0,
        help="gradient steps to train on heuristic buffer before self-play (default 0)",
    )
    parser.add_argument(
        "--heuristic-iterations",
        type=int,
        default=0,
        help="iterations of heuristic-only generate+train before self-play begins "
        "(default 0)",
    )
    parser.add_argument(
        "--greedy-warmup",
        action="store_true",
        help="start by playing AlphaZero vs GreedyAgent; auto-switch to self-play once "
        "rolling avg score exceeds --warmup-threshold",
    )
    parser.add_argument(
        "--warmup-threshold",
        type=float,
        default=20.0,
        help="avg AlphaZero score required to switch from warmup to self-play (default "
        "20)",
    )
    parser.add_argument(
        "--warmup-window",
        type=int,
        default=40,
        help="number of recent games used for the rolling score average (default 40)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="move selection temperature during self-play (default 1.0)",
    )
    parser.add_argument(
        "--random-eval-interval",
        type=int,
        default=0,
        help="evaluate vs random every N iterations (0 = never, default 0)",
    )
    parser.add_argument(
        "--value-only-iterations",
        type=int,
        default=0,
        help="train value head only for first N iterations before enabling policy "
        "training (default 0)",
    )
    parser.add_argument(
        "--skip-eval-iterations",
        type=int,
        default=0,
        help="skip evaluation for the first N iterations (default 0)",
    )
    parser.add_argument(
        "--clear-buffer-after-pretrain",
        action="store_true",
        help="clear the replay buffer after heuristic pretraining so self-play "
        "starts with a clean buffer (default: keep pretrain data mixed in)",
    )
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    log_path = setup_logging()
    logger.info("using device: %s", DEVICE)
    logger.info("log file: %s", log_path)
    logger.info("run parameters:")
    for key, value in sorted(vars(args).items()):
        logger.info("  %-24s %s", key, value)

    # ── Setup ──────────────────────────────────────────────────────────────
    net = AzulNet().to(DEVICE)
    if args.load:
        load_checkpoint(net, args.load)

    best_net = copy.deepcopy(net)
    generation = 0

    buf = ReplayBuffer(capacity=args.buffer_size)
    trainer = Trainer(net, lr=args.lr, batch_size=args.batch_size, device=DEVICE)

    warmup_mode = args.greedy_warmup
    greedy_opponent = GreedyAgent()
    recent_az_scores: list[float] = []
    iter_results: list[IterResult] = []

    if warmup_mode:
        logger.info(
            "greedy warmup enabled -- will switch to self-play once rolling avg "
            "score exceeds %.0f over %d games",
            args.warmup_threshold,
            args.warmup_window,
        )

    # ── Heuristic pretraining ──────────────────────────────────────────────
    if args.pretrain_games > 0:
        logger.info(
            "pretraining buffer with %d heuristic games...", args.pretrain_games
        )
        collect_heuristic_games(buf, num_games=args.pretrain_games)
        logger.info("pretraining complete -- buffer size: %d", len(buf))

    if args.pretrain_steps > 0 and len(buf) >= args.batch_size:
        logger.info(
            "pretraining network for %d steps on heuristic buffer...",
            args.pretrain_steps,
        )
        accum = _init_loss_accumulator()
        for step in range(1, args.pretrain_steps + 1):
            losses = trainer.train_step(buf, value_only=True)
            _accumulate_losses(accum, losses)
            if step % 500 == 0:
                logger.info(
                    "  pretrain step %d/%d -- %s",
                    step,
                    args.pretrain_steps,
                    _format_loss_line(accum, step),
                )
        logger.info(
            "pretraining complete -- %d steps -- %s",
            args.pretrain_steps,
            _format_loss_line(accum, args.pretrain_steps),
        )

    if args.heuristic_iterations > 0:
        logger.info(
            "running %d heuristic-only iterations (%d games each, %d steps each)...",
            args.heuristic_iterations,
            args.games_per_iter,
            args.train_steps,
        )
        for h_iter in range(1, args.heuristic_iterations + 1):
            collect_heuristic_games(buf, num_games=args.games_per_iter)
            if len(buf) < args.batch_size:
                logger.info(
                    "  heuristic iter %d -- buffer too small, skipping train", h_iter
                )
                continue
            accum = _init_loss_accumulator()
            for _ in range(args.train_steps):
                losses = trainer.train_step(buf, value_only=True)
                _accumulate_losses(accum, losses)
            avg_loss = accum["total"] / args.train_steps
            logger.info(
                "  heuristic iter %d/%d -- buffer size %d -- %s",
                h_iter,
                args.heuristic_iterations,
                len(buf),
                _format_loss_line(accum, args.train_steps),
            )
        logger.info("heuristic iterations complete")

    # ── Optionally clear the buffer before self-play ───────────────────────
    if args.clear_buffer_after_pretrain and len(buf) > 0:
        logger.info(
            "clearing replay buffer (was %d examples) before self-play starts",
            len(buf),
        )
        buf.clear()

    # ── Self-play loop ─────────────────────────────────────────────────────
    logger.info(
        "starting training -- %d iterations, %d games/iter, %d sims/move",
        args.iterations,
        args.games_per_iter,
        args.simulations,
    )

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        logger.info("-" * 60)
        logger.info("iteration %d / %d", iteration, args.iterations)

        # ── 1. Self-play data generation ───────────────────────────────────
        opponent = greedy_opponent if warmup_mode else None
        mode_label = "warmup" if warmup_mode else "self-play"
        logger.info("generating %d %s games...", args.games_per_iter, mode_label)
        net.eval()
        az_scores = collect_self_play(
            buf,
            net=net,
            num_games=args.games_per_iter,
            simulations=args.simulations,
            temperature=args.temperature,
            opponent=opponent,
            device=DEVICE,
        )
        logger.info("replay buffer size: %d", len(buf))

        # ── Rolling average and auto-switch ────────────────────────────────
        recent_az_scores.extend(az_scores)
        recent_az_scores = recent_az_scores[-args.warmup_window :]
        rolling_avg = sum(recent_az_scores) / len(recent_az_scores)
        logger.info(
            "AlphaZero rolling avg score (last %d games): %.1f",
            len(recent_az_scores),
            rolling_avg,
        )
        if warmup_mode and rolling_avg >= args.warmup_threshold:
            warmup_mode = False
            logger.info(
                "★ rolling avg %.1f >= %.0f -- switching to self-play mode",
                rolling_avg,
                args.warmup_threshold,
            )

        # ── 2. Training ────────────────────────────────────────────────────
        if len(buf) < args.batch_size:
            logger.info("buffer too small to train yet, skipping training step")
            continue

        value_only = iteration <= args.value_only_iterations
        if value_only and iteration == 1:
            logger.info(
                "value-only training for first %d iterations",
                args.value_only_iterations,
            )
        if not value_only and iteration == args.value_only_iterations + 1:
            logger.info("★ switching to full policy+value training")

        logger.info("running %d training steps...", args.train_steps)
        accum = _init_loss_accumulator()
        for _ in range(args.train_steps):
            losses = trainer.train_step(buf, value_only=value_only)
            _accumulate_losses(accum, losses)
        avg_loss = accum["total"] / args.train_steps
        logger.info(_format_loss_line(accum, args.train_steps))

        # ── 3. Evaluation ──────────────────────────────────────────────────
        if iteration <= args.skip_eval_iterations:
            logger.info(
                "skipping eval (iteration %d <= skip-eval-iterations %d)",
                iteration,
                args.skip_eval_iterations,
            )
            win_rate = 0.0
            promoted = False
        else:
            logger.info(
                "evaluating new net vs best net (%d games, %d sims)...",
                args.eval_games,
                args.eval_simulations,
            )
            win_rate = evaluate(
                net,
                best_net,
                num_games=args.eval_games,
                simulations=args.eval_simulations,
                win_threshold=args.win_threshold,
                buf=buf,
                record=True,
                iteration=iteration,
                generation=generation,
            )
            logger.info(f"new net win rate vs best: {win_rate * 100:.1f}%")

            if (
                args.random_eval_interval > 0
                and iteration % args.random_eval_interval == 0
            ):
                rng_wr = evaluate_vs_random(
                    net, num_games=20, simulations=args.eval_simulations
                )
                logger.info(f"new net win rate vs random: {rng_wr * 100:.1f}%")

        # ── 4. Keep if better ──────────────────────────────────────────────
        # ── 4. Keep if better ──────────────────────────────────────────────
        promoted = (iteration > args.skip_eval_iterations) and (
            win_rate >= args.win_threshold
        )
        if promoted:
            generation += 1
            best_net = copy.deepcopy(net)
            save_checkpoint(net, generation)
            logger.info("✓ new best model -- generation %d", generation)
        else:
            logger.info(
                "✗ new model did not beat threshold (%.0f%% < %.0f%%) -- keeping old "
                "best",
                win_rate * 100,
                args.win_threshold * 100,
            )
            net.load_state_dict(copy.deepcopy(best_net.state_dict()))
            trainer.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        elapsed = time.time() - t0
        logger.info("iteration time: %.1fs", elapsed)

        result = IterResult(
            iteration=iteration,
            mode=mode_label,
            avg_loss=avg_loss,
            win_rate=win_rate,
            promoted=promoted,
            generation=generation if promoted else 0,
            az_avg=rolling_avg,
            elapsed=elapsed,
        )
        iter_results.append(result)
        logger.info(_summary_line(result))

    # ── Final summary ──────────────────────────────────────────────────────
    print_summary(iter_results, generation)


if __name__ == "__main__":
    main()
