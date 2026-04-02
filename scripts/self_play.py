# scripts/self_play.py

"""Self-play harness — runs N games between two agents and logs statistics."""

import argparse
import logging
import sys
from dataclasses import dataclass
import time

from agents.base import Agent
from agents.cautious import CautiousAgent
from agents.efficient import EfficientAgent
from agents.greedy import GreedyAgent
from agents.mcts import MCTSAgent
from agents.random import RandomAgent
from engine.game import Game

logger = logging.getLogger(__name__)

AGENT_REGISTRY = {
    "random": RandomAgent,
    "cautious": CautiousAgent,
    "efficient": EfficientAgent,
    "greedy": GreedyAgent,
    "mcts": MCTSAgent,
}


# ── Result dataclasses ─────────────────────────────────────────────────────


@dataclass
class GameResult:
    """The outcome of a single game.

    Attributes:
        winner: Index of the winning player (0 or 1), or None for a tie.
        scores: Final scores for each player.
        rounds: Number of rounds played.
    """

    winner: int | None
    scores: list[int]
    rounds: int


@dataclass
class SeriesStats:
    """Aggregated statistics across a series of games.

    Attributes:
        total_games: Number of games played.
        win_rate_p1: Fraction of games won by player 1.
        win_rate_p2: Fraction of games won by player 2.
        tie_rate: Fraction of games that ended in a tie.
        avg_score_p1: Average final score for player 1.
        avg_score_p2: Average final score for player 2.
        avg_rounds: Average number of rounds per game.
    """

    total_games: int
    win_rate_p1: float
    win_rate_p2: float
    tie_rate: float
    avg_score_p1: float
    avg_score_p2: float
    avg_rounds: float


# ── Core functions ─────────────────────────────────────────────────────────


def run_game(p1: Agent, p2: Agent) -> GameResult:
    """Play one complete game between two agents and return the result.

    Args:
        p1: Agent for player 0.
        p2: Agent for player 1.

    Returns:
        A GameResult with winner, scores, and round count.
    """
    agents = [p1, p2]
    game = Game()
    game.setup_round()

    while not game.is_game_over():
        # Play out all moves in the current round
        while game.legal_moves():
            current = game.state.current_player
            move = agents[current].choose_move(game)
            game.make_move(move)

        game.score_round()

        if not game.is_game_over():
            game.setup_round()

    game.score_game()

    scores = [p.score for p in game.state.players]
    max_score = max(scores)
    winners = [i for i, s in enumerate(scores) if s == max_score]
    winner = winners[0] if len(winners) == 1 else None

    return GameResult(
        winner=winner,
        scores=scores,
        rounds=game.state.round,
    )


def run_series(p1: Agent, p2: Agent, n: int) -> SeriesStats:
    """Run N games between two agents and return aggregated statistics.

    Args:
        p1: Agent for player 0.
        p2: Agent for player 1.
        n: Number of games to play.

    Returns:
        A SeriesStats summarising the series.
    """
    results = []
    last_update = time.time()

    for i in range(n):
        results.append(run_game(p1, p2))

        now = time.time()
        if now - last_update >= 5.0:
            pct = (i + 1) / n * 100
            p1_wins = sum(1 for r in results if r.winner == 0)
            p2_wins = sum(1 for r in results if r.winner == 1)
            print(
                f"  {i + 1}/{n} games ({pct:.0f}%) — "
                f"P1 {p1_wins} W / P2 {p2_wins} W",
                flush=True,
            )
            last_update = now

    p1_wins = sum(1 for r in results if r.winner == 0)
    p2_wins = sum(1 for r in results if r.winner == 1)
    ties = sum(1 for r in results if r.winner is None)

    return SeriesStats(
        total_games=n,
        win_rate_p1=p1_wins / n,
        win_rate_p2=p2_wins / n,
        tie_rate=ties / n,
        avg_score_p1=sum(r.scores[0] for r in results) / n,
        avg_score_p2=sum(r.scores[1] for r in results) / n,
        avg_rounds=sum(r.rounds for r in results) / n,
    )


# ── CLI ────────────────────────────────────────────────────────────────────


def _build_agent(name: str) -> Agent:
    """Instantiate an agent by name, raising a clear error if unknown."""
    if name not in AGENT_REGISTRY:
        known = ", ".join(AGENT_REGISTRY)
        print(f"Unknown agent {name!r}. Known agents: {known}", file=sys.stderr)
        sys.exit(1)
    return AGENT_REGISTRY[name]()


def _configure_logging(log_file: str) -> None:
    """Set up logging to both stdout and a file."""
    fmt = "%(asctime)s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def main() -> None:
    """Entry point for the self-play CLI."""
    parser = argparse.ArgumentParser(description="Run Azul self-play games.")
    parser.add_argument(
        "--games", type=int, default=100, help="Number of games to play (default: 100)"
    )
    parser.add_argument(
        "--p1", default="random", help="Agent for player 1 (default: random)"
    )
    parser.add_argument(
        "--p2", default="random", help="Agent for player 2 (default: random)"
    )
    parser.add_argument(
        "--log", default="self_play.log", help="Log file path (default: self_play.log)"
    )
    args = parser.parse_args()

    _configure_logging(args.log)

    p1 = _build_agent(args.p1)
    p2 = _build_agent(args.p2)

    logger.info("Starting %d games: %s vs %s", args.games, args.p1, args.p2)

    stats = run_series(p1, p2, n=args.games)

    logger.info("Results after %d games:", stats.total_games)
    logger.info("  P1 win rate : %.1f%%", stats.win_rate_p1 * 100)
    logger.info("  P2 win rate : %.1f%%", stats.win_rate_p2 * 100)
    logger.info("  Tie rate    : %.1f%%", stats.tie_rate * 100)
    logger.info("  Avg score P1: %.1f", stats.avg_score_p1)
    logger.info("  Avg score P2: %.1f", stats.avg_score_p2)
    logger.info("  Avg rounds  : %.1f", stats.avg_rounds)
    logger.info("Results written to %s", args.log)


if __name__ == "__main__":
    main()
