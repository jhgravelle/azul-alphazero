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

# (name, label, factory)
AGENT_REGISTRY: list[tuple[str, str, Callable]] = [
    ("human", "Human", lambda **_: None),
    ("random", "Random Bot", lambda **_: RandomAgent()),
    ("cautious", "Cautious Bot", lambda **_: CautiousAgent()),
    ("efficient", "Efficient Bot", lambda **_: EfficientAgent()),
    ("greedy", "Greedy Bot", lambda **_: GreedyAgent()),
    ("mcts", "MCTS Bot", lambda **_: MCTSAgent()),
    ("minimax", "Minimax Bot", lambda **_: MinimaxAgent()),
    ("alphabeta", "AlphaBeta Bot", lambda **_: AlphaBetaAgent()),
]

AGENT_NAMES = [name for name, _, _ in AGENT_REGISTRY]
AGENT_LABELS = {name: label for name, label, _ in AGENT_REGISTRY}
_FACTORIES = {name: factory for name, _, factory in AGENT_REGISTRY}


def make_agent(name: str, **kwargs) -> Agent | None:
    if name not in _FACTORIES:
        raise ValueError(f"Unknown agent: {name!r}")
    return _FACTORIES[name](**kwargs)
