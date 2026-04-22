# scripts/benchmark_agents.py
"""Benchmark agent thinking time at various depth settings.

Usage:
    python -m scripts.benchmark_agents
    python -m scripts.benchmark_agents --agents minimax alphabeta --moves 50
"""

import argparse
import time

from agents.registry import make_agent
from engine.game import Game

_DEPTH_CONFIGS = [
    ("easy   d(2,3,7) t(20,10)", (2, 3, 7), (20, 10)),
    ("medium d(3,5,7) t(20,10)", (3, 5, 7), (20, 10)),
    ("hard   d(3,5,8) t(20,10)", (3, 5, 8), (20, 10)),
    ("minimax default", (2, 3, 4), (10, 7)),
]


def benchmark_config(agent_name: str, depths, thresholds, num_games: int = 5) -> dict:
    agent = make_agent(agent_name, depths=depths, thresholds=thresholds)
    if agent is None:
        raise ValueError(f"{agent_name!r} is not a bot")

    first_move_times = []
    first_move_nodes = []
    other_times = []
    other_nodes = []

    for _ in range(num_games):
        game = Game()
        game.setup_round()
        first_move = True

        while not game.is_game_over():
            t0 = time.perf_counter()
            move = agent.choose_move(game)
            elapsed = (time.perf_counter() - t0) * 1000
            nodes = getattr(agent, "_nodes", 0)

            if first_move:
                first_move_times.append(elapsed)
                first_move_nodes.append(nodes)
                first_move = False
            else:
                other_times.append(elapsed)
                other_nodes.append(nodes)

            game.make_move(move)
            game.advance()

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "first_ms": avg(first_move_times),
        "first_nodes": avg(first_move_nodes),
        "other_ms": avg(other_times),
        "other_nodes": avg(other_nodes),
        "overall": avg(first_move_times + other_times),
        "total_moves": len(first_move_times) + len(other_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark agent thinking time")
    parser.add_argument("--agents", nargs="+", default=["alphabeta"])  # "minimax",
    parser.add_argument(
        "--games",
        type=int,
        default=3,
        help="Number of games to time per config (default: 3)",
    )
    args = parser.parse_args()

    targets = [1000, 500, 200, 100, 50, 25, 10]

    print(f"\nBenchmarking {args.games} games per config")
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Target ms/move: {targets}")
    print("=" * 90)

    for agent_name in args.agents:
        print(f"\n{agent_name}:")
        print(
            f"  {'Config':<22} {'1st ms':>8} {'1st nodes':>10} "
            f"{'other ms':>9} {'other nodes':>12} {'overall':>8}"
        )
        print(f"  {'-'*22} {'-'*8} {'-'*10} " f"{'-'*9} {'-'*12} {'-'*8}")

        for label, depths, thresholds in _DEPTH_CONFIGS:
            r = benchmark_config(agent_name, depths, thresholds, num_games=args.games)
            print(
                f"  {label:<22} "
                f"{r['first_ms']:>7.0f}ms "
                f"{r['first_nodes']:>10.0f} "
                f"{r['other_ms']:>8.0f}ms "
                f"{r['other_nodes']:>12.0f} "
                f"{r['overall']:>7.0f}ms"
            )


if __name__ == "__main__":
    main()
