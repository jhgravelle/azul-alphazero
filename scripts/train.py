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
import time
from pathlib import Path

import torch

from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import Trainer, collect_self_play
from agents.alphazero import AlphaZeroAgent
from engine.game import Game
from agents.random import RandomAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints")


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(net: AzulNet, generation: int) -> Path:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"gen_{generation:04d}.pt"
    torch.save(net.state_dict(), path)
    logger.info("saved checkpoint → %s", path)
    return path


def load_checkpoint(net: AzulNet, path: str) -> None:
    net.load_state_dict(torch.load(path, map_location="cpu"))
    logger.info("loaded checkpoint ← %s", path)


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    new_net: AzulNet,
    old_net: AzulNet,
    num_games: int = 40,
    simulations: int = 50,
) -> float:
    """Play new vs old; return new_net win rate (ties count as 0.5)."""
    from engine.game import Game

    new_agent = AlphaZeroAgent(new_net, simulations=simulations, temperature=0.0)
    old_agent = AlphaZeroAgent(old_net, simulations=simulations, temperature=0.0)

    new_wins = 0.0
    for i in range(num_games):
        game = Game()
        game.setup_round()
        # Alternate who plays as player 0 to reduce first-mover bias
        agents = [new_agent, old_agent] if i % 2 == 0 else [old_agent, new_agent]
        new_is_p0 = i % 2 == 0

        while not game.is_game_over():
            if not game.legal_moves():
                break
            agent = agents[game.state.current_player]
            game.make_move(agent.choose_move(game))

        scores = [p.score for p in game.state.players]
        if scores[0] == scores[1]:
            new_wins += 0.5
        elif (scores[0] > scores[1]) == new_is_p0:
            new_wins += 1.0

    return new_wins / num_games


def evaluate_vs_random(
    net: AzulNet,
    num_games: int = 20,
    simulations: int = 50,
) -> float:
    """Quick sanity check: win rate vs RandomAgent."""
    az_agent = AlphaZeroAgent(net, simulations=simulations, temperature=0.0)
    rng_agent = RandomAgent()
    wins = 0.0
    for i in range(num_games):
        game = Game()
        game.setup_round()
        agents = [az_agent, rng_agent] if i % 2 == 0 else [rng_agent, az_agent]
        az_is_p0 = i % 2 == 0
        while not game.is_game_over():
            if not game.legal_moves():
                break
            game.make_move(agents[game.state.current_player].choose_move(game))
        scores = [p.score for p in game.state.players]
        if scores[0] == scores[1]:
            wins += 0.5
        elif (scores[0] > scores[1]) == az_is_p0:
            wins += 1.0
    return wins / num_games


# ── Main loop ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Azul AlphaZero training loop")
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="number of generate→train→eval cycles (default 30)",
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
        default=100,
        help="gradient steps per iteration (default 100)",
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
        default=50,
        help="MCTS simulations per move during evaluation (default 50)",
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
        default=0.55,
        help="new model win rate required to replace old (default 0.55)",
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
        "--temperature",
        type=float,
        default=1.0,
        help="move selection temperature during self-play (default 1.0)",
    )
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    net = AzulNet()
    if args.load:
        load_checkpoint(net, args.load)

    best_net = copy.deepcopy(net)  # the reigning champion
    generation = 0

    buf = ReplayBuffer(capacity=args.buffer_size)
    trainer = Trainer(net, lr=args.lr, batch_size=args.batch_size)

    logger.info(
        "starting training — %d iterations, %d games/iter, %d sims/move",
        args.iterations,
        args.games_per_iter,
        args.simulations,
    )

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        logger.info("═" * 60)
        logger.info("iteration %d / %d", iteration, args.iterations)

        # ── 1. Self-play data generation ──────────────────────────────────
        logger.info("generating %d self-play games…", args.games_per_iter)
        net.eval()
        collect_self_play(
            buf,
            net=net,
            num_games=args.games_per_iter,
            simulations=args.simulations,
            temperature=args.temperature,
        )
        logger.info("replay buffer size: %d", len(buf))

        # ── 2. Training ───────────────────────────────────────────────────
        if len(buf) < args.batch_size:
            logger.info("buffer too small to train yet, skipping training step")
            continue

        logger.info("running %d training steps…", args.train_steps)
        total_loss = 0.0
        for _ in range(args.train_steps):
            total_loss += trainer.train_step(buf)
        avg_loss = total_loss / args.train_steps
        logger.info("avg loss: %.4f", avg_loss)

        # ── 3. Evaluation ─────────────────────────────────────────────────
        logger.info(
            "evaluating new net vs best net (%d games, %d sims)…",
            args.eval_games,
            args.eval_simulations,
        )
        win_rate = evaluate(
            net,
            best_net,
            num_games=args.eval_games,
            simulations=args.eval_simulations,
        )
        logger.info("new net win rate vs best: %.1f%%", win_rate * 100)

        # Quick sanity: vs random
        rng_wr = evaluate_vs_random(
            net, num_games=20, simulations=args.eval_simulations
        )
        logger.info("new net win rate vs random: %.1f%%", rng_wr * 100)

        # ── 4. Keep if better ─────────────────────────────────────────────
        if win_rate >= args.win_threshold:
            generation += 1
            best_net = copy.deepcopy(net)
            save_checkpoint(net, generation)
            logger.info("✓ new best model — generation %d", generation)
        else:
            logger.info(
                "✗ new model did not beat threshold (%.0f%% < %.0f%%) — keeping old "
                "best",
                win_rate * 100,
                args.win_threshold * 100,
            )
            # Reset net weights to best — prevents diverging from a bad iteration
            net.load_state_dict(copy.deepcopy(best_net.state_dict()))
            trainer.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        elapsed = time.time() - t0
        logger.info("iteration time: %.1fs", elapsed)

    logger.info("═" * 60)
    logger.info("training complete — %d generations produced", generation)
    if generation > 0:
        logger.info("best checkpoint: checkpoints/gen_%04d.pt", generation)


if __name__ == "__main__":
    main()
