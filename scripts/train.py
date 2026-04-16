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
from neural.trainer import Trainer, collect_self_play, collect_heuristic_games
from agents.alphazero import AlphaZeroAgent
from agents.greedy import GreedyAgent
from agents.random import RandomAgent
from engine.game import Game

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
_MAX_MOVES = 2000


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


def save_eval_recording(
    new_net: AzulNet,
    old_net: AzulNet,
    iteration: int,
    generation: int,
    simulations: int,
) -> None:
    """Play and record one eval game between candidate and best net."""
    from engine.game_recorder import GameRecorder
    from neural.search_tree import SearchTree, make_policy_value_fn

    eval_dir = Path("recordings/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    nets = [new_net, old_net]

    recorder = GameRecorder(
        player_names=[f"candidate (iter {iteration})", f"best (gen {generation:04d})"],
        player_types=["alphazero", "alphazero"],
    )
    game = Game()
    game.setup_round()
    recorder.start_round(game)

    last_round = game.state.round
    moves = 0

    while not game.is_game_over() and moves < _MAX_MOVES:
        if not game.legal_moves():
            break
        current = game.state.current_player
        # Fresh tree each move — no stale state possible.
        tree = SearchTree(
            policy_value_fn=make_policy_value_fn(nets[current], DEVICE),
            simulations=simulations,
            temperature=0.0,
        )
        move = tree.choose_move(game)
        recorder.record_move(move, player_index=current)
        game.make_move(move)
        game.advance_round_if_needed()
        if not game.is_game_over() and game.state.round != last_round:
            last_round = game.state.round
            recorder.start_round(game)
        moves += 1

    recorder.finalize(game)
    filename = f"eval_iter_{iteration:03d}_vs_gen_{generation:04d}.json"
    recorder.save(eval_dir / filename)
    logger.info("saved eval recording -> %s", filename)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    new_net: AzulNet,
    old_net: AzulNet,
    num_games: int = 40,
    simulations: int = 25,
    win_threshold: float = 0.52,
) -> float:
    """Play new vs old; return new_net win rate (ties count as 0.5).

    Exits early if the outcome is already decided:
      - early pass: new_net has won enough to clear the threshold regardless
      - early fail: new_net has lost too many to clear the threshold
    """
    new_agent = AlphaZeroAgent(
        new_net, simulations=simulations, temperature=0.0, device=DEVICE
    )
    old_agent = AlphaZeroAgent(
        old_net, simulations=simulations, temperature=0.0, device=DEVICE
    )

    wins_needed = math.ceil(num_games * win_threshold)
    losses_allowed = num_games - wins_needed

    new_wins = 0.0
    losses = 0
    games_played = 0

    for i in range(num_games):
        game = Game()
        game.setup_round()
        agents = [new_agent, old_agent] if i % 2 == 0 else [old_agent, new_agent]
        new_is_p0 = i % 2 == 0
        moves = 0

        while not game.is_game_over() and moves < _MAX_MOVES:
            if not game.legal_moves():
                break
            agent = agents[game.state.current_player]
            game.make_move(agent.choose_move(game))
            game.advance_round_if_needed()
            moves += 1

        if moves >= _MAX_MOVES:
            logger.warning(
                "eval game %d hit move cap (%d) -- scoring as tie", i, _MAX_MOVES
            )

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

        logger.debug(
            "  eval game %d/%d -- scores %s -- %s (new win rate so far: %.0f%%)",
            i + 1,
            num_games,
            scores,
            result,
            new_wins / games_played * 100,
        )

        # Early exit checks
        if new_wins >= wins_needed:
            logger.info(
                "  eval early pass after %d/%d games -- %.0f wins >= %d needed",
                games_played,
                num_games,
                new_wins,
                wins_needed,
            )
            break
        if losses > losses_allowed:
            logger.info(
                "  eval early fail after %d/%d games -- %d losses > %d allowed",
                games_played,
                num_games,
                losses,
                losses_allowed,
            )
            break

    return new_wins / num_games


def evaluate_vs_random(
    net: AzulNet,
    num_games: int = 20,
    simulations: int = 25,
) -> float:
    """Quick sanity check: win rate vs RandomAgent."""
    az_agent = AlphaZeroAgent(
        net, simulations=simulations, temperature=0.0, device=DEVICE
    )
    rng_agent = RandomAgent()
    wins = 0.0
    for i in range(num_games):
        game = Game()
        game.setup_round()
        agents = [az_agent, rng_agent] if i % 2 == 0 else [rng_agent, az_agent]
        az_is_p0 = i % 2 == 0
        moves = 0
        while not game.is_game_over() and moves < _MAX_MOVES:
            if not game.legal_moves():
                break
            game.make_move(agents[game.state.current_player].choose_move(game))
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
        default=50_000,
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
        total_loss = 0.0
        for step in range(1, args.pretrain_steps + 1):
            total_loss += trainer.train_step(buf, value_only=True)["total"]
            if step % 500 == 0:
                logger.info(
                    "  pretrain step %d/%d -- avg loss: %.4f",
                    step,
                    args.pretrain_steps,
                    total_loss / step,
                )
        logger.info(
            "pretraining complete -- %d steps -- final avg loss: %.4f",
            args.pretrain_steps,
            total_loss / args.pretrain_steps,
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
            total_loss = 0.0
            for _ in range(args.train_steps):
                total_loss += trainer.train_step(buf, value_only=True)["total"]
            avg_loss = total_loss / args.train_steps
            logger.info(
                "  heuristic iter %d/%d -- buffer size %d -- avg loss: %.4f",
                h_iter,
                args.heuristic_iterations,
                len(buf),
                avg_loss,
            )
        logger.info("heuristic iterations complete")

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

        logger.info("running %d training steps...", args.train_steps)
        total_loss = 0.0
        for _ in range(args.train_steps):
            total_loss += trainer.train_step(buf, value_only=True)["total"]
        avg_loss = total_loss / args.train_steps
        logger.info("avg loss: %.4f", avg_loss)

        # ── 3. Evaluation ──────────────────────────────────────────────────
        logger.info(
            "evaluating new net vs best net (%d games, %d sims)...",
            args.eval_games,
            args.eval_simulations,
        )
        save_eval_recording(
            net,
            best_net,
            iteration=iteration,
            generation=generation,
            simulations=args.eval_simulations,
        )
        win_rate = evaluate(
            net,
            best_net,
            num_games=args.eval_games,
            simulations=args.eval_simulations,
            win_threshold=args.win_threshold,
        )
        logger.info("new net win rate vs best: %.1f%%", win_rate * 100)

        if args.random_eval_interval > 0 and iteration % args.random_eval_interval == 0:
            rng_wr = evaluate_vs_random(
                net, num_games=20, simulations=args.eval_simulations
            )
            logger.info("new net win rate vs random: %.1f%%", rng_wr * 100)

        # ── 4. Keep if better ──────────────────────────────────────────────
        promoted = win_rate >= args.win_threshold
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
