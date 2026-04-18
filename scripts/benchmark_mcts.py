# scripts/benchmark_mcts.py
"""Benchmark MCTS simulation speed across different configurations.

Usage:
    python -m scripts.benchmark_mcts
"""

from __future__ import annotations

import time
import torch
from engine.game import Game
from neural.model import AzulNet
from neural.search_tree import (
    SearchTree,
    make_policy_value_fn,
    make_batch_policy_value_fn,
)


def time_moves(
    net: AzulNet,
    simulations: int,
    num_moves: int,
    device: torch.device,
    batched: bool,
    batch_size: int | None = None,
) -> float:
    """Time num_moves individual move decisions and return seconds per move."""
    game = Game()
    game.setup_round()

    policy_fn = make_policy_value_fn(net, device=device)
    batch_fn = make_batch_policy_value_fn(net, device=device) if batched else None

    total = 0.0
    for _ in range(num_moves):
        g = game.clone()
        tree = SearchTree(
            policy_value_fn=policy_fn,
            batch_policy_value_fn=batch_fn,
            simulations=simulations,
            temperature=1.0,
            batch_size=batch_size or simulations,
        )
        t0 = time.perf_counter()
        tree.choose_move(g)
        total += time.perf_counter() - t0

    return total / num_moves


def main() -> None:
    net_cpu = AzulNet()
    cpu = torch.device("cpu")

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        net_gpu = AzulNet().to(torch.device("cuda"))
        gpu = torch.device("cuda")

    num_moves = 10
    simulations = 500

    print(f"\nMCTS benchmark — {simulations} simulations, {num_moves} moves each\n")
    print(f"{'Config':<35} {'sec/move':>10} {'moves/sec':>12}")
    print("-" * 60)

    configs = [
        ("Serial CPU", net_cpu, cpu, False, None),
        ("Batched CPU (batch=sims)", net_cpu, cpu, True, simulations),
        ("Batched CPU (batch=50)", net_cpu, cpu, True, 50),
        ("Batched CPU (batch=100)", net_cpu, cpu, True, 100),
    ]

    if has_cuda:
        configs += [
            ("Serial GPU", net_gpu, gpu, False, None),
            ("Batched GPU (batch=sims)", net_gpu, gpu, True, simulations),
            ("Batched GPU (batch=50)", net_gpu, gpu, True, 50),
            ("Batched GPU (batch=100)", net_gpu, gpu, True, 100),
        ]

    for label, net, device, batched, batch_size in configs:
        sec_per_move = time_moves(
            net=net,
            simulations=simulations,
            num_moves=num_moves,
            device=device,
            batched=batched,
            batch_size=batch_size,
        )
        print(f"{label:<35} {sec_per_move:>10.3f} {1/sec_per_move:>12.1f}")

    print()


if __name__ == "__main__":
    main()
