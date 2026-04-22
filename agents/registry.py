# agents/registry.py
"""Single source of truth for all available agents."""

from agents.alphabeta import AlphaBetaAgent
from agents.base import Agent
from agents.cautious import CautiousAgent
from agents.efficient import EfficientAgent
from agents.greedy import GreedyAgent
from agents.mcts import MCTSAgent
from agents.minimax import MinimaxAgent
from agents.random import RandomAgent
from typing import Callable

# (name, label, factory, hidden)
AGENT_REGISTRY: list[tuple[str, str, Callable, bool]] = [
    ("human", "Human", lambda **_: None, False),
    ("random", "Random Bot", lambda **_: RandomAgent(), False),
    ("cautious", "Cautious Bot", lambda **_: CautiousAgent(), False),
    ("efficient", "Efficient Bot", lambda **_: EfficientAgent(), False),
    ("greedy", "Greedy Bot", lambda **_: GreedyAgent(), False),
    ("mcts", "MCTS Bot", lambda **_: MCTSAgent(), False),
    ("minimax", "Minimax Bot", lambda **kwargs: MinimaxAgent(**kwargs), False),
    ("alphabeta", "AlphaBeta Bot", lambda **kwargs: AlphaBetaAgent(**kwargs), True),
    (
        "alphabeta_easy",
        "Easy Bot",
        lambda **_: AlphaBetaAgent(depths=(1, 2, 3), thresholds=(20, 10)),
        False,
    ),
    (
        "alphabeta_medium",
        "Easy Bot",
        lambda **_: AlphaBetaAgent(depths=(2, 3, 7), thresholds=(20, 10)),
        False,
    ),
    (
        "alphabeta_hard",
        "Medium Bot",
        lambda **_: AlphaBetaAgent(depths=(3, 5, 7), thresholds=(20, 10)),
        False,
    ),
    (
        "alphabeta_extreme",
        "Hard Bot",
        lambda **_: AlphaBetaAgent(depths=(4, 6, 8), thresholds=(20, 10)),
        False,
    ),
]

AGENT_NAMES = [name for name, _, _, _ in AGENT_REGISTRY]
AGENT_LABELS = {name: label for name, label, _, _ in AGENT_REGISTRY}
_FACTORIES = {name: factory for name, _, factory, _ in AGENT_REGISTRY}


def make_agent(name: str, **kwargs) -> Agent | None:
    if name not in _FACTORIES:
        raise ValueError(f"Unknown agent: {name!r}")
    return _FACTORIES[name](**kwargs)
