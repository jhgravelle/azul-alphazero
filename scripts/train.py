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
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import torch

from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import (
    AgentSpec,
    Trainer,
    collect_parallel,
    collect_ab_parallel,
    evaluate_parallel,
    _pretrain_matchups,
    collect_heuristic_parallel,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
RECORDINGS_DIR = Path("recordings/training")


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

    logging.getLogger("engine.game").setLevel(logging.WARNING)

    return log_path


logger = logging.getLogger(__name__)


# ── Iteration result ──────────────────────────────────────────────────────────


@dataclass
class IterResult:
    iteration: int
    mode: str
    avg_loss: float
    win_rate: float
    promoted: bool
    generation: int
    elapsed: float


# ── Loss accumulator helpers ──────────────────────────────────────────────────

_LOSS_KEYS = ("total", "policy", "value", "value_win", "value_diff", "value_abs")


def _init_loss_accumulator() -> dict[str, float]:
    return {k: 0.0 for k in _LOSS_KEYS}


def _accumulate_losses(accum: dict[str, float], step_losses: dict[str, float]) -> None:
    for k in _LOSS_KEYS:
        accum[k] += step_losses.get(k, 0.0)


def _format_loss_line(accum: dict[str, float], n_steps: int) -> str:
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


def save_latest_checkpoint(net: AzulNet, args: argparse.Namespace) -> None:
    import json

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(net.state_dict(), CHECKPOINT_DIR / "latest.pt")
    params_path = CHECKPOINT_DIR / "latest_params.json"
    params_path.write_text(json.dumps(vars(args), indent=2))


def load_checkpoint(net: AzulNet, path: str) -> None:
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    logger.info("loaded checkpoint <- %s", path)


# ── Agent spec helpers ────────────────────────────────────────────────────────


def _az_spec(net: AzulNet, simulations: int) -> AgentSpec:
    """Build an AlphaZero AgentSpec from a net, moving weights to CPU."""
    return AgentSpec(
        type="alphazero",
        state_dict={k: v.cpu() for k, v in net.state_dict().items()},
        simulations=simulations,
    )


# ── Showcase recording ────────────────────────────────────────────────────────


def _record_showcase_game(
    net: AzulNet,
    simulations: int,
    iteration: int,
    device: torch.device,
) -> None:
    """Play one greedy AZ vs AZ game and save it as a training recording.

    Not used for training — diagnostic only. Gives a human-readable view
    of how the current candidate net plays before eval runs.
    """
    from agents.alphazero import AlphaZeroAgent
    from engine.game import Game
    from engine.game_recorder import GameRecorder

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDINGS_DIR / f"iter_{iteration:04d}.json"

    net.eval()
    net.cpu()
    agent_0 = AlphaZeroAgent(net, simulations=simulations, temperature=0.0)
    agent_1 = AlphaZeroAgent(net, simulations=simulations, temperature=0.0)
    agents = [agent_0, agent_1]

    recorder = GameRecorder(
        player_names=[f"Iter {iteration}", f"Iter {iteration}"],
        player_types=["alphazero", "alphazero"],
    )

    game = Game()
    game.setup_round()
    recorder.start_round(game)
    prev_round = game.round

    while True:
        if not game.legal_moves():
            break
        agent = agents[game.current_player_index]
        move = agent.choose_move(game)
        recorder.record_move(move, game.current_player_index)
        game.make_move(move)
        game.advance()
        if game.is_game_over():
            break
        if game.round != prev_round:
            recorder.start_round(game)
            for a in agents:
                a.reset_tree(game)
            prev_round = game.round
        else:
            for a in agents:
                a.advance(move)

    recorder.finalize(game)
    recorder.save(path)
    net.to(device)
    scores = recorder.record.final_scores
    logger.info(
        "showcase game saved -> %s  [%d vs %d]",
        path,
        scores[0],
        scores[1],
    )


# ── Summary helpers ───────────────────────────────────────────────────────────


def _summary_line(r: IterResult) -> str:
    promoted = f"+ gen{r.generation:04d}" if r.promoted else "-      "
    return (
        f"iter {r.iteration:3d} | loss {r.avg_loss:.4f} | "
        f"win {r.win_rate * 100:5.1f}% {promoted} | "
        f"{r.mode} | {r.elapsed:.0f}s"
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


# ── Training steps helper ─────────────────────────────────────────────────────


def _run_training_steps(
    trainer: Trainer,
    buf: ReplayBuffer,
    num_steps: int,
    diff_only: bool,
    value_only: bool = False,
    log_interval: int = 500,
) -> tuple[float, dict[str, float]]:
    """Run num_steps training steps, log at intervals.

    Returns (avg_total_loss, full_accum_dict).
    """
    total_accum = _init_loss_accumulator()
    interval_accum = _init_loss_accumulator()

    for step in range(1, num_steps + 1):
        losses = trainer.train_step(buf, value_only=value_only, diff_only=diff_only)
        _accumulate_losses(total_accum, losses)
        _accumulate_losses(interval_accum, losses)
        if step % log_interval == 0:
            logger.info(
                "  step %d/%d -- %s",
                step,
                num_steps,
                _format_loss_line(interval_accum, log_interval),
            )
            interval_accum = _init_loss_accumulator()

    avg_loss = total_accum["total"] / num_steps if num_steps > 0 else 0.0
    return avg_loss, total_accum


# ── Main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Azul AlphaZero training loop")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=200,
        help="mirror pairs per iteration for both AZ collection and AB injection",
    )
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations per move (self-play and eval)",
    )
    parser.add_argument(
        "--eval-games", type=int, default=48, help="eval mirror pairs per iteration"
    )
    parser.add_argument("--win-threshold", type=float, default=0.55)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--load",
        type=str,
        default="checkpoints/latest.pt",
        help="checkpoint to resume from (skipped if file does not exist)",
    )
    parser.add_argument(
        "--initial-generation",
        type=int,
        default=0,
        help="generation counter starting value (set to match loaded checkpoint)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel worker processes for game generation and eval (default 8)",
    )
    parser.add_argument(
        "--diff-only",
        action="store_true",
        help="train score differential head only (Phase 1 calibration)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="fill buffer with ABeasy vs ABeasy games and train before "
        "starting the main iteration loop",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout probability on trunk output (0.0 to disable)",
    )
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    log_path = setup_logging()
    logger.info("using device: %s", DEVICE)
    logger.info("log file: %s", log_path)
    logger.info("run parameters:")
    for key, value in sorted(vars(args).items()):
        logger.info("  %-28s %s", key, value)

    # ── Setup ──────────────────────────────────────────────────────────────
    net = AzulNet(dropout=args.dropout).to(DEVICE)
    if args.load and Path(args.load).exists():
        load_checkpoint(net, args.load)
    elif args.load and not Path(args.load).exists():
        logger.info("checkpoint not found at %s -- starting fresh", args.load)

    best_net = copy.deepcopy(net)
    generation = args.initial_generation

    buf = ReplayBuffer(capacity=args.buffer_size)
    trainer = Trainer(net, lr=args.lr, batch_size=args.batch_size, device=DEVICE)

    # ── Pretrain phase (optional, single pass before iteration loop) ───────
    if args.pretrain:
        num_pretrain_pairs = args.buffer_size // 100
        num_pretrain_steps = num_pretrain_pairs * 20
        logger.info(
            "pretrain -- collecting %d ABeasy vs ABeasy pairs (%d workers)...",
            num_pretrain_pairs,
            args.workers,
        )
        collect_heuristic_parallel(
            buf,
            num_pairs=num_pretrain_pairs,
            matchups=_pretrain_matchups(),
            num_workers=args.workers,
        )
        logger.info("pretrain buffer size: %d", len(buf))

        if len(buf) >= args.batch_size:
            logger.info("pretrain -- running %d training steps...", num_pretrain_steps)
            avg_loss, accum = _run_training_steps(
                trainer, buf, num_pretrain_steps, diff_only=args.diff_only
            )
            logger.info(
                "pretrain complete -- %s",
                _format_loss_line(accum, num_pretrain_steps),
            )
            save_checkpoint(net, generation=0)
            best_net = copy.deepcopy(net)
            save_latest_checkpoint(net, args)
        else:
            logger.info("pretrain -- buffer too small to train, skipping training step")

    iter_results: list[IterResult] = []
    az_vs_az_mode = False
    ab_easy_spec = AgentSpec(type="alphabeta", depth=1, threshold=4)

    # ── Main iteration loop ────────────────────────────────────────────────
    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        logger.info("-" * 60)
        logger.info(
            "iteration %d / %d  [mode: %s]",
            iteration,
            args.iterations,
            "az-vs-az" if az_vs_az_mode else "az-vs-abeasy",
        )

        # ── 1. Data collection ─────────────────────────────────────────────
        net.eval()
        az_spec = _az_spec(net, args.simulations)

        if az_vs_az_mode:
            logger.info("generating %d AZ vs AZ mirror pairs...", args.games_per_iter)
            collect_parallel(
                buf,
                spec_0=az_spec,
                spec_1=az_spec,
                num_pairs=args.games_per_iter,
                num_workers=args.workers,
            )
        else:
            logger.info(
                "generating %d AZ vs ABeasy mirror pairs...", args.games_per_iter
            )
            stats = collect_parallel(
                buf,
                spec_0=az_spec,
                spec_1=ab_easy_spec,
                num_pairs=args.games_per_iter,
                num_workers=args.workers,
            )
            total_games = stats["wins_0"] + stats["wins_1"] + stats["ties"]
            az_win_rate = (
                (stats["wins_0"] + 0.5 * stats["ties"]) / total_games
                if total_games > 0
                else 0.0
            )
            logger.info(
                "AZ win rate vs ABeasy: %.1f%%  (%dW / %dL / %dT)",
                az_win_rate * 100,
                stats["wins_0"],
                stats["wins_1"],
                stats["ties"],
            )
            if az_win_rate >= args.win_threshold:
                logger.info(
                    "AZ win rate %.1f%% >= %.0f%% -- switching to az-vs-az",
                    az_win_rate * 100,
                    args.win_threshold * 100,
                )
                az_vs_az_mode = True

            logger.info(
                "generating %d ABeasy vs ABeasy mirror pairs...",
                args.games_per_iter,
            )
            collect_ab_parallel(
                buf,
                num_pairs=args.games_per_iter,
                num_workers=args.workers,
            )

        logger.info("buffer size after collection: %d", len(buf))

        # ── 2. Training ────────────────────────────────────────────────────
        if len(buf) < args.batch_size:
            logger.info("buffer too small to train yet, skipping")
            save_latest_checkpoint(net, args)
            continue

        net.train()
        logger.info("running %d training steps...", args.train_steps)
        avg_loss, accum = _run_training_steps(
            trainer,
            buf,
            args.train_steps,
            diff_only=args.diff_only,
        )
        logger.info(
            "training complete -- %s", _format_loss_line(accum, args.train_steps)
        )

        # ── 3. Showcase recording ──────────────────────────────────────────
        net.eval()
        logger.info("recording showcase game (iter %d)...", iteration)
        _record_showcase_game(net, args.simulations, iteration, DEVICE)

        # ── 4. Evaluation: new net vs best net ─────────────────────────────
        win_rate = 0.0
        promoted = False

        logger.info(
            "evaluating new net vs best net (%d pairs, %d sims)...",
            args.eval_games,
            args.simulations,
        )
        win_rate = evaluate_parallel(
            net,
            best_net,
            num_pairs=args.eval_games,
            simulations=args.simulations,
            buf=buf if az_vs_az_mode else None,
            num_workers=args.workers,
        )
        logger.info("new net win rate vs best: %.1f%%", win_rate * 100)

        # ── 5. Promote or reset ────────────────────────────────────────────
        if win_rate >= args.win_threshold:
            generation += 1
            best_net = copy.deepcopy(net)
            save_checkpoint(net, generation)
            logger.info("promoted -- new best model generation %d", generation)
            promoted = True
        else:
            logger.info(
                "did not beat threshold (%.1f%% < %.0f%%) -- resetting to best",
                win_rate * 100,
                args.win_threshold * 100,
            )
            net.load_state_dict(copy.deepcopy(best_net.state_dict()))
            trainer.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        elapsed = time.time() - t0
        logger.info("iteration time: %.1fs", elapsed)

        mode_label = "az-vs-az" if az_vs_az_mode else "az-vs-abeasy"
        result = IterResult(
            iteration=iteration,
            mode=mode_label,
            avg_loss=avg_loss,
            win_rate=win_rate,
            promoted=promoted,
            generation=generation if promoted else 0,
            elapsed=elapsed,
        )
        iter_results.append(result)
        logger.info(_summary_line(result))
        save_latest_checkpoint(net, args)

    print_summary(iter_results, generation)


if __name__ == "__main__":
    main()
