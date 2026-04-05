# agents/alphazero.py

"""AlphaZero-style agent: PUCT tree search guided by a neural network."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from agents.base import Agent
from agents.move_filters import non_floor_moves
from engine.game import Game, Move
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE
from neural.model import AzulNet

_PUCT_C = 1.5


@dataclass
class AZNode:
    """A node in the PUCT search tree."""

    game: Game
    parent: AZNode | None = None
    move: Move | None = None
    prior: float = 0.0

    visits: int = 0
    total_value: float = 0.0

    children: list[AZNode] = field(default_factory=list)

    # None = not yet initialised; [] = fully expanded (all moves tried)
    _untried_moves: list[Move] | None = None
    _untried_priors: list[float] | None = None

    # ── properties ────────────────────────────────────────────────────────

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    @property
    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return self._untried_moves is not None and len(self._untried_moves) == 0

    def puct_score(self, parent_visits: int) -> float:
        u = _PUCT_C * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q_value + u


class AlphaZeroAgent(Agent):
    """MCTS agent using PUCT selection and a neural network for evaluation.

    Uses lazy expansion — child nodes are created one at a time as they are
    selected rather than all at once, avoiding clone overhead for moves that
    are never visited.
    """

    def __init__(
        self,
        net: AzulNet,
        simulations: int = 200,
        temperature: float = 0.0,
        mask_floor: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.net = net
        self.simulations = simulations
        self.temperature = temperature
        self.mask_floor = mask_floor
        self.device = device

    # ── public API ────────────────────────────────────────────────────────

    def choose_move(self, game: Game) -> Move:
        root = AZNode(game=game.clone())
        self._expand(root)

        for _ in range(self.simulations):
            node = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        return self._pick_move(root)

    # def choose_move(self, game: Game) -> Move:
    #     import time

    #     root = AZNode(game=game.clone())
    #     self._expand(root)

    #     t_select = 0.0
    #     t_evaluate = 0.0
    #     t_backprop = 0.0

    #     for _ in range(self.simulations):
    #         t0 = time.perf_counter()
    #         node = self._select(root)
    #         t1 = time.perf_counter()
    #         value = self._evaluate(node)
    #         t2 = time.perf_counter()
    #         self._backpropagate(node, value)
    #         t3 = time.perf_counter()

    #         t_select += t1 - t0
    #         t_evaluate += t2 - t1
    #         t_backprop += t3 - t2

    #     total = t_select + t_evaluate + t_backprop
    #     import logging

    #     logging.getLogger(__name__).debug(
    #         "MCTS breakdown (%d sims) -- "
    #         "select+expand: %.1f%% evaluate: %.1f%% backprop: %.1f%%",
    #         self.simulations,
    #         t_select / total * 100,
    #         t_evaluate / total * 100,
    #         t_backprop / total * 100,
    #     )

    #     return self._pick_move(root)

    def get_policy_targets(self, game: Game) -> tuple[Move, list[tuple[Move, float]]]:
        """Run simulations and return (chosen_move, [(move, visit_fraction), ...])."""
        root = AZNode(game=game.clone())
        self._expand(root)

        for _ in range(self.simulations):
            node = self._select(root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        total_visits = sum(c.visits for c in root.children)
        policy = [
            (c.move, c.visits / total_visits if total_visits else 0.0)
            for c in root.children
            if c.move is not None
        ]
        return self._pick_move(root), policy

    # ── MCTS steps ────────────────────────────────────────────────────────

    def _select(self, node: AZNode) -> AZNode:
        """Descend the tree, creating one new child when an untried move exists."""
        while not node.is_terminal:
            if node._untried_moves is None:
                # Node has never been expanded — evaluate it directly.
                return node
            if node._untried_moves:
                # Pop one untried move and create its child now.
                move = node._untried_moves.pop()
                prior = node._untried_priors.pop()  # type: ignore[union-attr]
                child_game = node.game.clone()
                child_game.make_move(move)
                child = AZNode(
                    game=child_game,
                    parent=node,
                    move=move,
                    prior=prior,
                )
                node.children.append(child)
                return child
            # Fully expanded — descend by PUCT score.
            node = max(node.children, key=lambda c: c.puct_score(node.visits))
        return node

    def _expand(self, node: AZNode) -> None:
        """Initialise the untried move list and precompute policy priors.

        Does not create any child nodes — children are created lazily by
        _select on each simulation visit.
        """
        if node._untried_moves is not None:
            return

        legal = node.game.legal_moves()
        if not legal:
            node._untried_moves = []
            node._untried_priors = []
            return

        if self.mask_floor:
            legal = non_floor_moves(legal)

        priors = self._policy_priors(node.game, legal)
        node._untried_moves = list(legal)
        node._untried_priors = list(priors)

    def _evaluate(self, node: AZNode) -> float:
        """Ask the value head for a position estimate.

        Returns a value in (-1, 1) from the current player's perspective.
        If the game is over, return the actual outcome.
        """
        if node.is_terminal:
            return self._terminal_value(node.game)
        # Expand on first evaluation so the node is ready for future visits.
        if node._untried_moves is None:
            self._expand(node)
        state = encode_state(node.game).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            _, value = self.net(state)
        return value.item()

    def _backpropagate(self, node: AZNode, value: float) -> None:
        """Walk back to root, flipping value sign at each player boundary."""
        current: AZNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            value = -value
            current = current.parent

    # ── helpers ───────────────────────────────────────────────────────────

    def _policy_priors(self, game: Game, legal: list[Move]) -> list[float]:
        """Return softmax probabilities over legal moves from the policy head."""
        state = encode_state(game).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(state)
        logits = logits.squeeze(0)
        mask = torch.full((MOVE_SPACE_SIZE,), float("-inf"), device=self.device)
        for move in legal:
            idx = encode_move(move, game)
            mask[idx] = logits[idx]
        probs = F.softmax(mask, dim=0)
        return [probs[encode_move(m, game)].item() for m in legal]

    def _pick_move(self, root: AZNode) -> Move:
        """Select a move from the root's children based on temperature."""
        if not root.children:
            return root.game.legal_moves()[0]
        if self.temperature == 0.0:
            best = max(root.children, key=lambda c: c.visits)
            assert best.move is not None
            return best.move
        visits = torch.tensor(
            [c.visits ** (1.0 / self.temperature) for c in root.children],
            dtype=torch.float32,
        )
        probs = visits / visits.sum()
        idx = int(torch.multinomial(probs, num_samples=1).item())
        chosen = root.children[idx].move
        assert chosen is not None
        return chosen

    def _terminal_value(self, game: Game) -> float:
        """Return +1 / -1 / 0 from the current player's perspective."""
        scores = [p.score for p in game.state.players]
        if scores[0] > scores[1]:
            val = 1.0
        elif scores[1] > scores[0]:
            val = -1.0
        else:
            val = 0.0
        if game.state.current_player == 1:
            val = -val
        return val
