# neural/search_tree.py
"""Game-owned MCTS search tree with transposition table and subtree reuse.

The SearchTree owns:
    - The Zobrist table (one instance, reused across rounds)
    - The transposition table (hash → AZNode)
    - The root node (advances as moves are made)

Agents are stateless policy/value providers. The tree handles all
MCTS bookkeeping: selection, expansion, backpropagation, subtree reuse.

Factory canonicalization
------------------------
When expanding a node, factories with identical tile multisets are treated
as a single source. This collapses redundant branches without changing the
game engine.

Round boundaries as leaf nodes
-------------------------------
When a simulation reaches the end of a round (all factories and center
empty, round not yet scored), it stops and evaluates with the value head
rather than crossing into the next round's random factory setup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable
import torch
from engine.game import Game, Move, CENTER
from neural.model import AzulNet
from neural.zobrist import ZobristTable

_PUCT_C = 1.5

# Shared Zobrist table — one instance for the lifetime of the process.
_ZOBRIST = ZobristTable()


def make_policy_value_fn(
    net: "AzulNet",
    device: "torch.device | None" = None,
) -> "PolicyValueFn":
    """Build a policy/value function backed by an AzulNet.

    This is the standard way to create a PolicyValueFn for AlphaZeroAgent.
    A uniform/zero function is available for testing by passing net=None
    via the existing make_policy_value_fn in test_search_tree.py.
    """
    import torch
    import torch.nn.functional as F
    from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE

    if device is None:
        device = torch.device("cpu")

    def fn(game: "Game", legal: "list[Move]") -> "tuple[list[float], float]":
        spatial, flat = encode_state(game)
        spatial = spatial.unsqueeze(0).to(device)
        flat = flat.unsqueeze(0).to(device)
        net.eval()
        with torch.no_grad():
            logits, value = net(spatial, flat)
        if not legal:
            return [], value.item()
        logits = logits.squeeze(0)
        mask = torch.full((MOVE_SPACE_SIZE,), float("-inf"), device=device)
        for move in legal:
            idx = encode_move(move, game)
            mask[idx] = logits[idx]
        probs = F.softmax(mask, dim=0)
        priors = [probs[encode_move(m, game)].item() for m in legal]
        return priors, value.item()

    return fn


# ── Node ──────────────────────────────────────────────────────────────────────


@dataclass
class AZNode:
    """A node in the PUCT search tree."""

    game: Game
    zobrist_hash: int = 0
    parent: AZNode | None = None
    move: Move | None = None
    prior: float = 0.0

    visits: int = 0
    total_value: float = 0.0

    children: list[AZNode] = field(default_factory=list)

    # None = not yet expanded; [] = fully expanded
    _untried_moves: list[Move] | None = None
    _untried_priors: list[float] | None = None

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    @property
    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    @property
    def is_round_boundary(self) -> bool:
        """True when all tiles have been taken but the round is not yet scored.

        This is a leaf node — we evaluate here rather than crossing into the
        next round's random factory setup.
        """
        state = self.game.state
        factories_empty = all(len(f) == 0 for f in state.factories)
        non_fp_center = [t for t in state.center if t.name != "FIRST_PLAYER"]
        center_empty = len(non_fp_center) == 0
        return factories_empty and center_empty and not self.game.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return self._untried_moves is not None and len(self._untried_moves) == 0

    def puct_score(self, parent_visits: int) -> float:
        u = _PUCT_C * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q_value + u


# ── Policy/value callable type ────────────────────────────────────────────────

# A function that takes a Game and a list of legal moves, and returns
# (priors, value) where priors is a list of floats parallel to legal_moves.
PolicyValueFn = Callable[[Game, list[Move]], tuple[list[float], float]]


# ── SearchTree ────────────────────────────────────────────────────────────────


class SearchTree:
    """MCTS search tree with transposition table and subtree reuse.

    Args:
        policy_value_fn: Callable that returns (priors, value) for a position.
        simulations:     Number of MCTS simulations per move.
        temperature:     Move selection temperature. 0.0 = greedy (most visits).
    """

    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        simulations: int = 200,
        temperature: float = 0.0,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.simulations = simulations
        self.temperature = temperature

        self._root: AZNode | None = None
        # Transposition table: zobrist_hash → AZNode
        self._table: dict[int, AZNode] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self, game: Game) -> None:
        """Start a fresh tree for a new round. Call at the start of each round."""
        self._table = {}
        self._root = self._make_node(game.clone(), parent=None, move=None, prior=0.0)

    def choose_move(self, game: Game) -> Move:
        """Run simulations from the current root and return the best move."""
        if self._root is None:
            self.reset(game)
        assert self._root is not None
        self._ensure_expanded(self._root)
        for _ in range(self.simulations):
            node = self._select(self._root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        return self._pick_move(self._root)

    def advance(self, move: Move) -> None:
        """Carry the subtree forward after a move is made.

        Finds the child matching the move (if it exists), makes it the new
        root, and prunes all siblings. Updates the transposition table.
        """
        if self._root is None:
            return

        # Find matching child if already created
        new_root = None
        for child in self._root.children:
            if child.move == move:
                new_root = child
                break

        if new_root is None:
            # Move was never explored — build a fresh node
            new_game = self._root.game.clone()
            new_game.make_move(move)
            # Do not advance round — round boundaries are leaf nodes
            new_root = self._make_node(new_game, parent=None, move=move, prior=0.0)
        else:
            new_root.parent = None

        # Prune siblings from transposition table
        for child in self._root.children:
            if child is not new_root:
                self._prune(child)

        self._root = new_root

    def get_policy_targets(self, game: Game) -> tuple[Move, list[tuple[Move, float]]]:
        """Run simulations and return (chosen_move, [(move, visit_fraction)])."""
        if self._root is None:
            self.reset(game)
        assert self._root is not None
        self._ensure_expanded(self._root)
        for _ in range(self.simulations):
            node = self._select(self._root)
            value = self._evaluate(node)
            self._backpropagate(node, value)

        total_visits = sum(c.visits for c in self._root.children)
        policy = [
            (c.move, c.visits / total_visits if total_visits else 0.0)
            for c in self._root.children
            if c.move is not None
        ]
        return self._pick_move(self._root), policy

    # ── Tree operations ────────────────────────────────────────────────────

    def _select(self, node: AZNode) -> AZNode:
        """Descend the tree, creating one new child when an untried move exists."""
        while not node.is_terminal and not node.is_round_boundary:
            if node._untried_moves is None:
                return node  # not yet expanded
            if node._untried_moves:
                # Create one new child from the next untried move
                move = node._untried_moves.pop()
                prior = node._untried_priors.pop()  # type: ignore[union-attr]

                child_game = node.game.clone()
                child_game.make_move(move)
                # Do NOT advance round — round boundary = leaf node

                child = self._make_node(child_game, parent=node, move=move, prior=prior)
                node.children.append(child)
                return child
            # Fully expanded — pick best PUCT child
            node = max(node.children, key=lambda c: c.puct_score(node.visits))
        return node

    def _ensure_expanded(self, node: AZNode) -> None:
        """Expand a node if not already expanded."""
        if node._untried_moves is not None:
            return
        if node.is_terminal or node.is_round_boundary:
            node._untried_moves = []
            node._untried_priors = []
            return
        legal = self._canonical_moves(node.game)
        if not legal:
            node._untried_moves = []
            node._untried_priors = []
            return
        priors, _ = self.policy_value_fn(node.game, legal)
        node._untried_moves = list(legal)
        node._untried_priors = list(priors)

    def _evaluate(self, node: AZNode) -> float:
        """Return a value estimate in (-1, 1) from the current player's perspective."""
        if node.is_terminal:
            return self._terminal_value(node.game)
        if node.is_round_boundary:
            _, value = self.policy_value_fn(node.game, [])
            return value
        if node._untried_moves is None:
            self._ensure_expanded(node)
        _, value = self.policy_value_fn(node.game, self._canonical_moves(node.game))
        return value

    def _backpropagate(self, node: AZNode, value: float) -> None:
        """Walk back to root, flipping sign at each player boundary."""
        current: AZNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            value = -value
            current = current.parent

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_node(
        self,
        game: Game,
        parent: AZNode | None,
        move: Move | None,
        prior: float,
    ) -> AZNode:
        """Create a node, register it in the transposition table."""
        h = _ZOBRIST.hash_state(game)
        # Check transposition table — reuse existing node if available
        if h in self._table:
            existing = self._table[h]
            # Update parent link to the closer ancestor (shallower path wins)
            if parent is not None and existing.parent is not None:
                existing.parent = parent
            return existing
        node = AZNode(
            game=game,
            zobrist_hash=h,
            parent=parent,
            move=move,
            prior=prior,
        )
        self._table[h] = node
        return node

    def _prune(self, node: AZNode) -> None:
        """Recursively remove a subtree from the transposition table."""
        self._table.pop(node.zobrist_hash, None)
        for child in node.children:
            self._prune(child)

    def _canonical_moves(self, game: Game) -> list[Move]:
        """Return legal moves with duplicate-factory moves collapsed.

        If two factories have identical tile multisets, only moves from the
        first (by sorted order) are included. This reduces redundant branches
        without changing the game engine.
        """
        legal = game.legal_moves()
        if not legal:
            return legal

        # Build a map from factory_idx → canonical factory_idx
        # Two factories are identical if their sorted tile lists match.
        factories = game.state.factories
        seen: dict[tuple, int] = {}  # multiset_key → first factory_idx with that key
        canonical: dict[int, int] = {}  # factory_idx → canonical_factory_idx

        for f_idx, factory in enumerate(factories):
            key = tuple(sorted(t.name for t in factory))
            if key in seen:
                canonical[f_idx] = seen[key]
            else:
                seen[key] = f_idx
                canonical[f_idx] = f_idx

        # Keep only moves from canonical factories (or CENTER)
        result = []
        for move in legal:
            if move.source == CENTER:
                result.append(move)
            elif canonical.get(move.source, move.source) == move.source:
                result.append(move)

        return result

    def _pick_move(self, root: AZNode) -> Move:
        """Select a move from root's children based on temperature."""
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
