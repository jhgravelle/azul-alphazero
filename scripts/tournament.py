# scripts/tournament.py
"""Run a round-robin tournament between agents and print win rates.

Usage:
    python -m scripts.tournament
    python -m scripts.tournament --agents greedy minimax --games 20
    python -m scripts.tournament --agents greedy minimax --games 50 --workers 8
    python -m scripts.tournament --agents random cautious greedy minimax --games 200

Available agents: random, cautious, efficient, greedy, mcts, minimax, alphabeta
"""

import argparse
import itertools
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents.base import Agent
from agents.registry import make_agent
from engine.game import Game


def _make_agent(name: str, depths=None, thresholds=None) -> Agent:
    kwargs = {}
    if depths is not None:
        kwargs["depths"] = tuple(depths)
    if thresholds is not None:
        kwargs["thresholds"] = tuple(thresholds)
    agent = make_agent(name, **kwargs)
    if agent is None:
        raise ValueError(f"Agent {name!r} cannot be used in tournament")
    return agent


def _play_one_game(
    name0: str,
    name1: str,
    game_index: int,
    depths0=None,
    thresholds0=None,
    depths1=None,
    thresholds1=None,
) -> dict:
    if game_index % 2 == 0:
        agents = [
            _make_agent(name0, depths0, thresholds0),
            _make_agent(name1, depths1, thresholds1),
        ]
        player_names = [name0, name1]
    else:
        agents = [
            _make_agent(name1, depths1, thresholds1),
            _make_agent(name0, depths0, thresholds0),
        ]
        player_names = [name1, name0]

    game = Game()
    game.setup_round()
    moves = 0
    think_times = [0.0, 0.0]
    move_counts = [0, 0]

    while not game.is_game_over():
        current = game.state.current_player
        t0 = time.perf_counter()
        move = agents[current].choose_move(game)
        think_times[current] += time.perf_counter() - t0
        move_counts[current] += 1
        game.make_move(move)
        game.advance()
        moves += 1
        if moves > 200:
            break

    scores = [p.score for p in game.state.players]
    if scores[0] == scores[1]:
        winner_name = None
    elif scores[0] > scores[1]:
        winner_name = player_names[0]
    else:
        winner_name = player_names[1]

    return {
        "winner": winner_name,
        "score_0": scores[0],
        "score_1": scores[1],
        "player_names": player_names,
        "think_times": think_times,
        "move_counts": move_counts,
    }


def run_matchup(
    name0: str,
    name1: str,
    games: int,
    workers: int,
    depths0=None,
    thresholds0=None,
    depths1=None,
    thresholds1=None,
) -> dict:
    wins = {name0: 0, name1: 0}
    ties = 0
    score_totals = {name0: 0, name1: 0}
    think_totals = {name0: 0.0, name1: 0.0}
    move_totals = {name0: 0, name1: 0}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _play_one_game,
                name0,
                name1,
                game_index,
                depths0,
                thresholds0,
                depths1,
                thresholds1,
            )
            for game_index in range(games)
        ]
        try:
            for future in as_completed(futures):
                result = future.result()
                player_names = result["player_names"]
                score_totals[player_names[0]] += result["score_0"]
                score_totals[player_names[1]] += result["score_1"]
                think_totals[player_names[0]] += result["think_times"][0]
                think_totals[player_names[1]] += result["think_times"][1]
                move_totals[player_names[0]] += result["move_counts"][0]
                move_totals[player_names[1]] += result["move_counts"][1]
                if result["winner"] is None:
                    ties += 1
                else:
                    wins[result["winner"]] += 1
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            raise

    return {
        "wins_0": wins[name0],
        "wins_1": wins[name1],
        "ties": ties,
        "avg_score_0": score_totals[name0] / games,
        "avg_score_1": score_totals[name1] / games,
        "avg_ms_per_move_0": (
            think_totals[name0] / move_totals[name0] * 1000 if move_totals[name0] else 0
        ),
        "avg_ms_per_move_1": (
            think_totals[name1] / move_totals[name1] * 1000 if move_totals[name1] else 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent tournament")
    parser.add_argument("--agents", nargs="+", default=["greedy", "minimax"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--depths0",
        type=int,
        nargs=3,
        default=None,
        metavar=("HIGH", "MID", "LOW"),
        help="depths tuple for first agent (e.g. --depths0 2 3 5)",
    )
    parser.add_argument(
        "--thresholds0",
        type=int,
        nargs=2,
        default=None,
        metavar=("T1", "T2"),
        help="thresholds tuple for first agent (e.g. --thresholds0 10 7)",
    )
    parser.add_argument(
        "--depths1",
        type=int,
        nargs=3,
        default=None,
        metavar=("HIGH", "MID", "LOW"),
        help="depths tuple for second agent (e.g. --depths1 2 3 6)",
    )
    parser.add_argument(
        "--thresholds1",
        type=int,
        nargs=2,
        default=None,
        metavar=("T1", "T2"),
        help="thresholds tuple for second agent (e.g. --thresholds1 10 7)",
    )
    args = parser.parse_args()

    matchups = list(itertools.combinations(args.agents, 2))

    print(
        f"\nTournament: {args.games} games per matchup, "
        f"workers={args.workers or 'auto'}"
    )
    print(f"Agents: {', '.join(args.agents)}")
    if args.depths0:
        print(f"  {args.agents[0]} depths={args.depths0} thresholds={args.thresholds0}")
    if args.depths1:
        print(
            f"  {args.agents[1] if len(args.agents) > 1 else ''} depths={args.depths1} "
            f"thresholds={args.thresholds1}"
        )
    print("=" * 60)

    overall_wins: dict[str, int] = defaultdict(int)
    overall_games: dict[str, int] = defaultdict(int)

    try:
        for i, (name0, name1) in enumerate(matchups):
            # Apply overrides only to the specific agent positions
            d0 = (
                args.depths0
                if name0 == args.agents[0]
                else args.depths1 if name0 == args.agents[1] else None
            )
            t0 = (
                args.thresholds0
                if name0 == args.agents[0]
                else args.thresholds1 if name0 == args.agents[1] else None
            )
            d1 = (
                args.depths0
                if name1 == args.agents[0]
                else args.depths1 if name1 == args.agents[1] else None
            )
            t1 = (
                args.thresholds0
                if name1 == args.agents[0]
                else args.thresholds1 if name1 == args.agents[1] else None
            )

            print(f"\n{name0} vs {name1} ({args.games} games)...", flush=True)
            result = run_matchup(name0, name1, args.games, args.workers, d0, t0, d1, t1)

            win_rate_0 = result["wins_0"] / args.games * 100
            win_rate_1 = result["wins_1"] / args.games * 100

            print(
                f"  {name0}: {result['wins_0']}W  {win_rate_0:.0f}%  "
                f"avg {result['avg_score_0']:.1f}pts  "
                f"{result['avg_ms_per_move_0']:.1f}ms/move"
            )
            print(
                f"  {name1}: {result['wins_1']}W  {win_rate_1:.0f}%  "
                f"avg {result['avg_score_1']:.1f}pts  "
                f"{result['avg_ms_per_move_1']:.1f}ms/move"
            )
            print(f"  Ties: {result['ties']}")

            overall_wins[name0] += result["wins_0"]
            overall_wins[name1] += result["wins_1"]
            overall_games[name0] += args.games
            overall_games[name1] += args.games
    except KeyboardInterrupt:
        print("\nTournament interrupted.")

    if len(args.agents) > 2:
        print("\n" + "=" * 60)
        print("Overall standings:")
        standings = sorted(
            args.agents,
            key=lambda n: overall_wins[n] / overall_games[n] if overall_games[n] else 0,
            reverse=True,
        )
        for name in standings:
            rate = (
                overall_wins[name] / overall_games[name] * 100
                if overall_games[name]
                else 0
            )
            print(
                f"  {name}: {overall_wins[name]}W / "
                f"{overall_games[name]}G  ({rate:.0f}%)"
            )


if __name__ == "__main__":
    main()
