# agents/alphazero.py
"""AlphaZero-style agent: thin wrapper around SearchTree + AzulNet."""

from __future__ import annotations

import torch

from agents.base import Agent
from engine.game import Game, Move
from neural.model import AzulNet
from neural.search_tree import SearchTree, make_policy_value_fn


class AlphaZeroAgent(Agent):
    """MCTS agent using PUCT selection and a neural network for evaluation.

    The agent owns a SearchTree for use in self-play and evaluation contexts
    where no external tree is provided. When the API passes in a shared tree
    via choose_move(game, tree=...), that tree is used instead and the
    agent's internal tree is ignored.
    """

    def __init__(
        self,
        net: AzulNet,
        simulations: int = 500,
        temperature: float = 0.0,
        device: torch.device = torch.device("cpu"),
        batched: bool = True,
    ) -> None:
        self.net = net
        self.simulations = simulations
        self.temperature = temperature
        self.device = device
        self.batched = batched

        from neural.search_tree import make_batch_policy_value_fn

        self._tree = SearchTree(
            policy_value_fn=make_policy_value_fn(net, device=device),
            batch_policy_value_fn=(
                make_batch_policy_value_fn(net, device=device) if batched else None
            ),
            simulations=simulations,
            temperature=temperature,
            batch_size=simulations,  # batch all sims in one forward pass
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def choose_move(self, game: Game, tree: SearchTree | None = None) -> Move:
        t = tree if tree is not None else self._tree
        if t._root is None or not t._root.game.legal_moves():
            t.reset(game)
        return t.choose_move(game)

    def get_policy_targets(
        self,
        game: Game,
        tree: SearchTree | None = None,
    ) -> tuple[Move, list[tuple[Move, float]]]:
        t = tree if tree is not None else self._tree
        if t._root is None or not t._root.game.legal_moves():
            t.reset(game)
        return t.get_policy_targets(game)

    def advance(self, move: Move, tree: SearchTree | None = None) -> None:
        """Carry the tree forward after a move is made."""
        t = tree if tree is not None else self._tree
        t.advance(move)

    def reset_tree(self, game: Game, tree: SearchTree | None = None) -> None:
        """Reset the tree for a new round."""
        t = tree if tree is not None else self._tree
        t.reset(game)
