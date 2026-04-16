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
import threading
from concurrent.futures import ThreadPoolExecutor
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
    virtual_loss: int = 0

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
        # Virtual loss: add pessimistic -1 per virtual visit to discourage
        # other threads from selecting the same path before real backprop.
        adj_visits = self.visits + self.virtual_loss
        if adj_visits == 0:
            q = 0.0
        else:
            q = (self.total_value - float(self.virtual_loss)) / adj_visits
        u = _PUCT_C * self.prior * math.sqrt(parent_visits) / (1 + adj_visits)
        return q + u


# ── Policy/value callable type ────────────────────────────────────────────────

# A function that takes a Game and a list of legal moves, and returns
# (priors, value) where priors is a list of floats parallel to legal_moves.
PolicyValueFn = Callable[[Game, list[Move]], tuple[list[float], float]]

# A batched variant: takes a list of (game, legal_moves) pairs and returns
# a parallel list of (priors, value) results — one net forward pass for all.
BatchPolicyValueFn = Callable[
    [list[tuple[Game, list[Move]]]],
    list[tuple[list[float], float]],
]


def make_batch_policy_value_fn(
    net: "AzulNet",
    device: "torch.device | None" = None,
) -> "BatchPolicyValueFn":
    """Build a batched policy/value function backed by an AzulNet.

    Encodes all positions in the batch, runs a single forward pass, and
    returns (priors, value) for each position.  Use this together with
    SearchTree(batch_policy_value_fn=...) to enable batched MCTS.
    """
    import torch
    import torch.nn.functional as F
    from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE

    if device is None:
        device = torch.device("cpu")

    def fn(
        batch: list[tuple[Game, list[Move]]],
    ) -> list[tuple[list[float], float]]:
        if not batch:
            return []

        spatials = []
        flats = []
        for game, _ in batch:
            spatial, flat = encode_state(game)
            spatials.append(spatial)
            flats.append(flat)

        spatial_t = torch.stack(spatials).to(device)  # (B, 12, 5, 6)
        flat_t = torch.stack(flats).to(device)  # (B, 47)

        net.eval()
        with torch.no_grad():
            logits_b, values_b = net(spatial_t, flat_t)

        results: list[tuple[list[float], float]] = []
        for i, (game, legal) in enumerate(batch):
            value = values_b[i].item()
            if not legal:
                results.append(([], value))
                continue
            logits = logits_b[i]
            mask = torch.full((MOVE_SPACE_SIZE,), float("-inf"), device=device)
            for move in legal:
                idx = encode_move(move, game)
                mask[idx] = logits[idx]
            probs = F.softmax(mask, dim=0)
            priors = [probs[encode_move(m, game)].item() for m in legal]
            results.append((priors, value))

        return results

    return fn


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
        n_threads: int = 1,
        batch_size: int = 8,
        batch_policy_value_fn: BatchPolicyValueFn | None = None,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.simulations = simulations
        self.temperature = temperature
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.batch_policy_value_fn = batch_policy_value_fn

        self._root: AZNode | None = None
        # Transposition table: zobrist_hash → AZNode
        self._table: dict[int, AZNode] = {}
        # Lock for tree mutations in batched/multi-threaded mode
        self._lock = threading.Lock()

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
        self._run_simulations()
        return self._pick_move(self._root)

    def advance(self, move: Move) -> None:
        if self._root is None:
            return

        new_root = None
        for child in self._root.children:
            if child.move == move:
                new_root = child
                break

        # Discard the child if it was treated as a boundary/terminal during
        # simulation (fully expanded but no children = never actually explored).
        if (
            new_root is not None
            and new_root.children == []
            and new_root._untried_moves == []
        ):
            new_root = None

        if new_root is None:
            new_game = self._root.game.clone()
            new_game.make_move(move)
            new_game.advance_round_if_needed()
            # Remove any stale transposition entry before making a new node
            old_hash = _ZOBRIST.hash_state(new_game)
            self._table.pop(old_hash, None)
            new_root = self._make_node(new_game, parent=None, move=move, prior=0.0)
        else:
            new_root.parent = None

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
        self._run_simulations()

        total_visits = sum(c.visits for c in self._root.children)
        policy = [
            (c.move, c.visits / total_visits if total_visits else 0.0)
            for c in self._root.children
            if c.move is not None
        ]
        return self._pick_move(self._root), policy

    # ── Simulation dispatch ────────────────────────────────────────────────

    def _run_simulations(self) -> None:
        """Run all simulations, dispatching to batched or single-threaded mode."""
        if self.batch_policy_value_fn is not None:
            self._run_batched_simulations()
        else:
            assert self._root is not None
            for _ in range(self.simulations):
                node = self._select(self._root)
                value = self._evaluate(node)
                self._backpropagate(node, value)

    def _run_batched_simulations(self) -> None:
        """Run simulations in batches with virtual loss for diverse leaf selection.

        Phase 1 — Select batch_size leaves sequentially under the lock, applying
                   virtual loss to each path so later selections diverge.
        Phase 2 — Evaluate all leaves in a single batched net forward pass
                   (no lock held — this is the expensive GPU step).
        Phase 3 — Undo virtual loss, expand leaves with returned priors, and
                   backpropagate values, each under the lock.

        n_threads controls the ThreadPoolExecutor used for Phase 3 backprop;
        for the current simple implementation Phase 1 is sequential (the lock
        serialises selection) and Phase 2 is always single-threaded (one GPU
        call).  True parallel selection is a future improvement.
        """
        assert self._root is not None
        assert self.batch_policy_value_fn is not None

        remaining = self.simulations
        while remaining > 0:
            batch_n = min(self.batch_size, remaining)
            remaining -= batch_n

            # ── Phase 1: select leaves under lock ─────────────────────────
            leaves_and_legals: list[tuple[AZNode, list[Move]]] = []
            for _ in range(batch_n):
                with self._lock:
                    node = self._select_vl(self._root)
                    if node.is_terminal or node.is_round_boundary:
                        legal: list[Move] = []
                    else:
                        legal = self._canonical_moves(node.game)
                    leaves_and_legals.append((node, legal))

            # ── Phase 2: batch evaluate (no lock) ─────────────────────────
            batch_results = self.batch_policy_value_fn(
                [(node.game, legal) for node, legal in leaves_and_legals]
            )

            # ── Phase 3: expand + backpropagate under lock ─────────────────
            def _backprop_one(
                node: AZNode,
                legal: list[Move],
                priors: list[float],
                value: float,
            ) -> None:
                with self._lock:
                    self._undo_vl(node)
                    if node.is_terminal:
                        value = self._terminal_value(node.game)
                    elif node._untried_moves is None:
                        # First visit — expand with priors from batch
                        if legal:
                            node._untried_moves = list(legal)
                            node._untried_priors = list(priors)
                        else:
                            node._untried_moves = []
                            node._untried_priors = []
                    self._backpropagate(node, value)

            with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
                futs = [
                    pool.submit(_backprop_one, node, legal, priors, value)
                    for (node, legal), (priors, value) in zip(
                        leaves_and_legals, batch_results
                    )
                ]
                for f in futs:
                    f.result()  # propagate any exceptions

    # ── Virtual loss helpers ───────────────────────────────────────────────

    def _select_vl(self, root: AZNode) -> AZNode:
        """Like _select but applies +1 virtual loss to every node on the path.

        Called under self._lock.  Virtual loss makes the chosen path look
        pessimistic to subsequent selections in the same batch, encouraging
        diversity without full thread synchronisation.
        """
        node = root
        while not node.is_terminal and not node.is_round_boundary:
            if node._untried_moves is None:
                # Unexpanded leaf — stop here
                node.virtual_loss += 1
                return node
            if node._untried_moves:
                # Expand one new child (priors already set on parent)
                move = node._untried_moves.pop()
                prior = node._untried_priors.pop()  # type: ignore[union-attr]
                child_game = node.game.clone()
                child_game.make_move(move)
                child = self._make_node(child_game, parent=node, move=move, prior=prior)
                node.children.append(child)
                node.virtual_loss += 1
                child.virtual_loss += 1
                return child
            # Fully expanded — pick best PUCT child (scores include VL)
            node.virtual_loss += 1
            node = max(node.children, key=lambda c: c.puct_score(node.visits))
        node.virtual_loss += 1
        return node

    def _undo_vl(self, node: AZNode) -> None:
        """Walk up the parent chain removing one virtual loss from each node.

        Called under self._lock after backpropagation is complete.
        """
        current: AZNode | None = node
        while current is not None:
            if current.virtual_loss > 0:
                current.virtual_loss -= 1
            current = current.parent

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
            legal = root.game.legal_moves()
            if not legal:
                raise RuntimeError(
                    "_pick_move called on terminal/boundary node with no children"
                )
            return legal[0]
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
