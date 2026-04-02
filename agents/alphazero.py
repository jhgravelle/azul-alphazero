# agents/alphazero.py

"""AlphaZero-style agent: PUCT tree search guided by a neural network."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from agents.base import Agent
from engine.game import Game, Move
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE
from neural.model import AzulNet


_PUCT_C = 1.5  # exploration constant — higher = more exploration


@dataclass
class AZNode:
    """A node in the PUCT search tree."""

    game: Game  # full game state at this node
    parent: AZNode | None = None
    move: Move | None = None  # move that led here from parent
    prior: float = 0.0  # P(move | parent state) from policy head

    visits: int = 0
    total_value: float = 0.0  # sum of value estimates (current player POV)

    children: list[AZNode] = field(default_factory=list)
    _untried_moves: list[Move] | None = None  # None = not yet expanded

    # ── properties ────────────────────────────────────────────────────────

    @property
    def q_value(self) -> float:
        """Mean value estimate; 0 if unvisited."""
        return self.total_value / self.visits if self.visits else 0.0

    @property
    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return self._untried_moves is not None and len(self._untried_moves) == 0

    # ── PUCT score ────────────────────────────────────────────────────────

    def puct_score(self, parent_visits: int) -> float:
        """PUCT = Q + C * P * sqrt(N_parent) / (1 + N_child)."""
        u = _PUCT_C * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q_value + u


class AlphaZeroAgent(Agent):
    """MCTS agent using PUCT selection and a neural network for evaluation.

    Parameters
    ----------
    net:
        Trained (or randomly initialised) AzulNet instance.
    simulations:
        Number of MCTS simulations per move.
    temperature:
        Controls move selection from root visit counts.
        0.0 = greedy (most-visited), 1.0 = proportional to visit counts.
    """

    def __init__(
        self,
        net: AzulNet,
        simulations: int = 200,
        temperature: float = 0.0,
    ) -> None:
        self.net = net
        self.simulations = simulations
        self.temperature = temperature

    # ── public API ────────────────────────────────────────────────────────

    def choose_move(self, game: Game) -> Move:
        root = AZNode(game=copy.deepcopy(game))
        self._expand(root)

        for _ in range(self.simulations):
            node = self._select(root)
            if not node.is_terminal:
                self._expand(node)
                # After expansion, evaluate one of the new children if any exist
                if node.children:
                    node = node.children[0]
            value = self._evaluate(node)
            self._backpropagate(node, value)

        return self._pick_move(root)

    def get_policy_targets(self, game: Game) -> tuple[Move, list[tuple[Move, float]]]:
        """Run simulations and return (chosen_move, [(move, visit_fraction), ...]).

        Used by collect_self_play() to produce policy targets for training.
        """
        root = AZNode(game=copy.deepcopy(game))
        self._expand(root)

        for _ in range(self.simulations):
            node = self._select(root)
            if not node.is_terminal:
                self._expand(node)
                if node.children:
                    node = node.children[0]
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
        """Descend the tree, picking the highest PUCT child at each level."""
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            node = max(node.children, key=lambda c: c.puct_score(node.visits))
        return node

    def _expand(self, node: AZNode) -> None:
        """Create children for all legal moves; set priors from the policy head."""
        if node._untried_moves is not None:
            return  # already expanded

        legal = node.game.legal_moves()
        if not legal:
            node._untried_moves = []
            return

        # Get policy priors from network
        priors = self._policy_priors(node.game, legal)

        node._untried_moves = []  # mark as expanded
        for move, prior in zip(legal, priors):
            child_game = copy.deepcopy(node.game)
            child_game.make_move(move)
            child = AZNode(
                game=child_game,
                parent=node,
                move=move,
                prior=prior,
            )
            node.children.append(child)

    def _evaluate(self, node: AZNode) -> float:
        """Ask the value head for a position estimate.

        Returns a value in (-1, 1) from the current player's perspective.
        If the game is over, return the actual outcome.
        """
        if node.is_terminal:
            return self._terminal_value(node.game)

        state = encode_state(node.game).unsqueeze(0)  # (1, STATE_SIZE)
        self.net.eval()
        with torch.no_grad():
            _, value = self.net(state)
        return value.item()  # scalar in (-1, 1)

    def _backpropagate(self, node: AZNode, value: float) -> None:
        """Walk back to root, flipping value sign at each player boundary."""
        current: AZNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            value = -value  # opponent's gain is our loss
            current = current.parent

    # ── helpers ───────────────────────────────────────────────────────────

    def _policy_priors(self, game: Game, legal: list[Move]) -> list[float]:
        """Return softmax probabilities over legal moves from the policy head."""
        state = encode_state(game).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            logits, _ = self.net(state)  # (1, MOVE_SPACE_SIZE)
        logits = logits.squeeze(0)  # (MOVE_SPACE_SIZE,)

        # Mask illegal moves
        mask = torch.full((MOVE_SPACE_SIZE,), float("-inf"))
        for move in legal:
            idx = encode_move(move, game)
            mask[idx] = logits[idx]

        probs = F.softmax(mask, dim=0)
        return [probs[encode_move(m, game)].item() for m in legal]

    def _pick_move(self, root: AZNode) -> Move:
        """Select a move from the root's children based on temperature."""
        if not root.children:
            return root.game.legal_moves()[0]  # fallback (shouldn't happen)

        if self.temperature == 0.0:
            best = max(root.children, key=lambda c: c.visits)
            assert best.move is not None
            return best.move

        # Sample proportional to visits^(1/temperature)
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
        """Return +1 / -1 / 0 from player 0's perspective."""
        scores = [p.score for p in game.state.players]
        if scores[0] > scores[1]:
            val = 1.0
        elif scores[1] > scores[0]:
            val = -1.0
        else:
            val = 0.0
        # Flip if current player is player 1
        if game.state.current_player == 1:
            val = -val
        return val
