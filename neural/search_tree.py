# neural/search_tree.py
"""PUCT-based MCTS search tree with transposition table and subtree reuse.

This is the neural network-backed search used by AlphaZeroAgent. It differs
from the plain MCTSAgent in several important ways:

  - Uses PUCT (Predictor + UCT) instead of UCB1. PUCT incorporates a learned
    prior probability from the policy head to guide exploration.
  - Values are in [-1.0, 1.0] (not [0.0, 1.0]) and flip sign at each
    backpropagation step, so positive always means "good for current player."
  - Supports batched inference so multiple leaf nodes can be evaluated by the
    neural network in parallel.
  - Maintains a transposition table (Zobrist hash → node) to reuse subtrees
    across moves via SearchTree.advance().
  - Uses virtual loss during batched search to discourage threads from all
    selecting the same path before any backpropagation has occurred.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable
import torch
from engine.game import Game, Move, CENTER, FLOOR
from neural.model import AzulNet
from neural.trainer import _SCORE_DIFF_DIVISOR
from neural.zobrist import ZobristTable

_PUCT_C = 1.0

_ZOBRIST = ZobristTable()

# After this many moves in a game, switch from high-temperature (exploratory)
# to low-temperature (decisive) move selection during self-play.
EXPLORATION_MOVES = 25
EXPLORATION_TEMP = 1.0
DETERMINISTIC_TEMP = 0.1


def make_policy_value_fn(
    net: "AzulNet",
    device: "torch.device | None" = None,
) -> "PolicyValueFn":
    """Return a single-game policy/value function backed by AzulNet.

    The returned callable encodes one game state, runs a forward pass, and
    returns (priors, value_diff) where value_diff is in [-1.0, 1.0] from
    the current player's perspective.

    Args:
        net:    Trained AzulNet instance.
        device: Torch device to run inference on. Defaults to CPU.
    """
    import torch
    from neural.encoder import encode_state, priors_from_3head

    if device is None:
        device = torch.device("cpu")

    def fn(game: "Game", legal: "list[Move]") -> "tuple[list[float], float]":
        encoding = encode_state(game)
        encoding = encoding.unsqueeze(0).to(device)
        net.eval()
        with torch.no_grad():
            (
                (src_logits, tile_logits, dst_logits),
                _value_win,
                value_diff,
                _value_abs,
            ) = net(encoding)
        if not legal:
            return [], value_diff.item()
        priors = priors_from_3head(
            src_logits.squeeze(0),
            tile_logits.squeeze(0),
            dst_logits.squeeze(0),
            legal,
            game,
        )
        return priors, value_diff.item()

    return fn


# ── Node ──────────────────────────────────────────────────────────────────────


@dataclass
class AZNode:
    """A node in the PUCT search tree.

    Each node represents a game state reached by a specific move from its
    parent. Values are stored from the current player's perspective at that
    node: positive means good for the player whose turn it is.

    Attributes:
        game:            The game state at this node (owned by the node).
        zobrist_hash:    Zobrist hash of game, used as the transposition table key.
        parent:          Parent node (None for the root).
        move:            The move applied to reach this node (None for the root).
        prior:           Policy prior probability assigned by the neural network.
        visits:          Total number of times this node has been visited.
        total_value:     Sum of backed-up values from the current player's perspective.
        virtual_loss:    Temporary penalty added during batched selection to discourage
                         multiple threads from selecting the same path before any
                         backpropagation has occurred. Reset to zero during backprop.
        children:        Child nodes expanded so far.
        _untried_moves:  Moves not yet expanded into children. None means the node
                         has not been evaluated by the policy network yet (unvisited).
                         An empty list means all moves have been expanded.
        _untried_priors: Policy priors corresponding to _untried_moves, in the same
                         order.
        _explored:       True once the entire subtree below this node is exhausted
                         (all paths lead to terminal or round-boundary nodes).
        net_value_diff:  Raw value-head output at this node, stored for the inspector
                         UI.
    """

    game: Game
    zobrist_hash: int = 0
    parent: AZNode | None = None
    move: Move | None = None
    prior: float = 0.0

    visits: int = 0
    total_value: float = 0.0
    virtual_loss: int = 0

    children: list[AZNode] = field(default_factory=list)

    _untried_moves: list[Move] | None = None
    _untried_priors: list[float] | None = None
    _explored: bool = False
    net_value_diff: float | None = None

    @property
    def q_value(self) -> float:
        """Average backed-up value from the current player's perspective."""
        return self.total_value / self.visits if self.visits else 0.0

    @property
    def is_terminal(self) -> bool:
        """True when the game is over (no more rounds to play)."""
        return self.game.is_game_over()

    @property
    def is_round_boundary(self) -> bool:
        """True when all tiles have been taken from factories but the round has not
        yet been scored. The search treats round boundaries as leaf nodes because
        advancing past them requires scoring and setup, which are handled outside
        the tree."""
        return self.game.is_round_over() and not self.game.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        """True when every legal move has been expanded into a child node.

        Distinct from _fully_explored: a node can be fully expanded but still
        have children with unexplored subtrees below them.
        """
        return self._untried_moves is not None and len(self._untried_moves) == 0

    @property
    def _fully_explored(self) -> bool:
        """True when the entire subtree rooted here has been exhausted.

        A fully explored subtree contains no nodes that could benefit from
        additional simulations — every path leads to a terminal, round boundary,
        or has been explicitly marked explored.
        """
        if self.is_terminal or self.is_round_boundary:
            return True
        return self._explored

    def _check_and_mark_explored(self) -> bool:
        """Mark this node as fully explored if all conditions are met.

        Called after every backpropagation pass. Propagation up the tree is
        handled by the backprop loop itself — each ancestor calls this method
        after updating its own counts.

        Returns:
            True if the node is now marked explored, False otherwise.
        """
        if self._explored:
            return True
        if self.is_terminal or self.is_round_boundary:
            self._explored = True
            return True
        if not self.is_fully_expanded:
            return False
        if not self.children:
            # All moves were expanded, but none produced children (shouldn't
            # happen in valid Azul, but handle it defensively).
            self._explored = True
            return True
        if all(child._fully_explored for child in self.children):
            self._explored = True
            return True
        return False

    def puct_score(self, parent_visits: int, unvisited_exploitation: float) -> float:
        """Compute the PUCT score used for child selection during tree descent.

        PUCT = Q + U, where:
          Q = exploitation term: average value adjusted for virtual losses.
              Unvisited nodes use the parent's net_value_diff as a prior estimate
              rather than 0.0, so they are not artificially suppressed before any
              evidence is collected.
          U = exploration term: scaled by prior and inversely by visit count.
              C must remain > 0 to break ties on the initial visit via the prior.

        Fully explored nodes return -infinity so they are never selected again.
        Virtual loss makes this node look artificially bad during batched
        selection, discouraging other threads from selecting the same path.

        Args:
            parent_visits:          Visit count of the parent node, used to scale U.
            unvisited_exploitation: Parent's net_value_diff, used as the exploitation
                                    estimate for unvisited nodes instead of 0.0.
        """
        if self._fully_explored:
            return float("-inf")
        adjusted_visits = self.visits + self.virtual_loss
        if adjusted_visits == 0:
            exploitation = unvisited_exploitation
        else:
            exploitation = (
                -(self.total_value + float(self.virtual_loss)) / adjusted_visits
            )
        exploration = (
            _PUCT_C * self.prior * math.sqrt(parent_visits) / (1 + adjusted_visits)
        )
        return exploitation + exploration


# ── Policy/value callable types ───────────────────────────────────────────────

PolicyValueFn = Callable[[Game, list[Move]], tuple[list[float], float]]
"""Single-game policy/value function. Given a game state and legal moves,
returns (priors, value) where value is in [-1.0, 1.0] from current player's view."""

BatchPolicyValueFn = Callable[
    [list[tuple[Game, list[Move]]]],
    list[tuple[list[float], float]],
]
"""Batched policy/value function. Evaluates a list of (game, legal_moves) pairs
in one forward pass for efficiency. Returns one (priors, value) per input."""


def make_batch_policy_value_fn(
    net: "AzulNet",
    device: "torch.device | None" = None,
) -> "BatchPolicyValueFn":
    """Return a batched policy/value function backed by AzulNet.

    Encodes all game states in a batch together and runs a single forward pass,
    which is significantly faster than calling the single-game version repeatedly
    when a GPU is available.

    Args:
        net:    Trained AzulNet instance.
        device: Torch device to run inference on. Defaults to CPU.
    """
    import torch
    from neural.encoder import encode_state, priors_from_3head

    if device is None:
        device = torch.device("cpu")

    def fn(
        batch: list[tuple[Game, list[Move]]],
    ) -> list[tuple[list[float], float]]:
        if not batch:
            return []

        encodings = [encode_state(game) for game, _ in batch]
        encoding_tensor = torch.stack(encodings).to(device)

        net.eval()
        with torch.no_grad():
            (
                (src_batch, tile_batch, dst_batch),
                _wins_batch,
                values_batch,
                _abs_batch,
            ) = net(encoding_tensor)

        results: list[tuple[list[float], float]] = []
        for index, (game, legal) in enumerate(batch):
            value = values_batch[index].item()
            if not legal:
                results.append(([], value))
                continue
            priors = priors_from_3head(
                src_batch[index], tile_batch[index], dst_batch[index], legal, game
            )
            results.append((priors, value))

        return results

    return fn


# ── SearchTree ────────────────────────────────────────────────────────────────


class SearchTree:
    """PUCT-based MCTS search tree with transposition table and subtree reuse.

    Workflow per move:
      1. Call choose_move(game) or get_policy_targets(game).
      2. Call advance(move) after the move is committed, to reuse the subtree.
      3. Repeat from step 1 for the next move.

    The tree can operate in two modes:
      - Single-threaded: uses policy_value_fn for sequential leaf evaluation.
      - Batched: uses batch_policy_value_fn to evaluate multiple leaves in one
        forward pass, dispatched via a ThreadPoolExecutor for backpropagation.

    Args:
        policy_value_fn:       Single-game policy/value callable, always required.
        simulations:           Number of simulations per move.
        temperature:           Move selection temperature. 0.0 = deterministic (most
                               visits).
                               Non-zero applies the 25-move temperature rule.
        n_threads:             Number of worker threads for batched backpropagation.
        batch_size:            Number of leaves to evaluate per batched forward pass.
        batch_policy_value_fn: Optional batched policy/value callable. When provided,
                               batched mode is used instead of single-threaded.
        use_heuristic_value:   If True, score round-boundary nodes with the heuristic
                               terminal_value function instead of the neural net.
    """

    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        simulations: int = 200,
        temperature: float = 0.0,
        n_threads: int = 1,
        batch_size: int = 8,
        batch_policy_value_fn: BatchPolicyValueFn | None = None,
        use_heuristic_value: bool = False,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.simulations = simulations
        self.temperature = temperature
        self.n_threads = n_threads
        self.batch_size = batch_size
        self.batch_policy_value_fn = batch_policy_value_fn
        self.use_heuristic_value = use_heuristic_value
        self._stable_batches: int = 0
        self._last_top_k: list[int] = []
        self._root: AZNode | None = None
        self._table: dict[int, AZNode] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────

    def reset(self, game: Game) -> None:
        """Discard the current tree and start fresh from the given game state."""
        self._table = {}
        self._root = self._make_node(game.clone(), parent=None, move=None, prior=0.0)
        self._stable_batches = 0
        self._last_top_k = []

    def choose_move(self, game: Game) -> Move:
        """Run simulations from the current root and return the best move.

        If no root exists, one is created from the given game state. The game
        passed in is never modified.
        """
        if self._root is None:
            self.reset(game)
        assert self._root is not None
        self._ensure_expanded(self._root)
        self._run_simulations()
        return self._pick_move(self._root)

    def advance(self, move: Move) -> None:
        """Advance the tree root to the child corresponding to move.

        Reuses the existing subtree for that child (if it exists and has been
        explored), discarding all other branches. If the child is unexplored or
        the move was never in the tree, builds a fresh root for the post-move state.
        """
        if self._root is None:
            return

        new_root = self._find_child_for_move(move)

        if self._child_is_unexplored(new_root):
            new_root = self._build_fresh_root(move)
        else:
            assert new_root is not None
            new_root.parent = None

        self._prune_siblings(new_root)
        self._root = new_root

    def get_policy_targets(self, game: Game) -> tuple[Move, list[tuple[Move, float]]]:
        """Run simulations and return the chosen move and visit-count policy targets.

        Policy targets are (move, visit_fraction) pairs normalized over all children.
        Used during self-play to generate training data for the policy head.
        """
        if self._root is None:
            self.reset(game)
        assert self._root is not None
        self._ensure_expanded(self._root)
        self._run_simulations()

        total_visits = sum(child.visits for child in self._root.children)
        policy = [
            (child.move, child.visits / total_visits if total_visits else 0.0)
            for child in self._root.children
            if child.move is not None
        ]
        return self._pick_move(self._root), policy

    # ── Advance helpers ────────────────────────────────────────────────────

    def _find_child_for_move(self, move: Move) -> AZNode | None:
        """Return the existing child node for move, or None if not found."""
        assert self._root is not None
        for child in self._root.children:
            if child.move == move:
                return child
        return None

    def _child_is_unexplored(self, child: AZNode | None) -> bool:
        """Return True if the child has no search results worth reusing."""
        return child is None or (child.children == [] and child._untried_moves == [])

    def _build_fresh_root(self, move: Move) -> AZNode:
        """Create a new root node by applying move to the current root's game."""
        assert self._root is not None
        new_game = self._root.game.clone()
        new_game.make_move(move)
        new_game.advance()
        old_hash = _ZOBRIST.hash_state(new_game)
        self._table.pop(old_hash, None)
        return self._make_node(new_game, parent=None, move=move, prior=0.0)

    def _prune_siblings(self, new_root: AZNode | None) -> None:
        """Prune all children of the old root except new_root."""
        assert self._root is not None
        for child in self._root.children:
            if child is not new_root:
                self._prune(child)

    # ── Simulation dispatch ────────────────────────────────────────────────

    def _run_simulations(self) -> None:
        """Dispatch simulations in single-threaded or batched mode."""
        if self._root is not None and self._root._fully_explored:
            return
        if self.batch_policy_value_fn is not None:
            self._run_batched_simulations()
        else:
            self._run_sequential_simulations()

    def _run_sequential_simulations(self) -> None:
        """Run simulations one at a time using the single-game policy/value function."""
        assert self._root is not None
        for _ in range(self.simulations):
            if self._root._fully_explored:
                break
            leaf = self._select(self._root)
            value = self._evaluate(leaf)
            self._backpropagate(leaf, value)

    def _run_batched_simulations(self) -> None:
        """Run simulations in batches, evaluating multiple leaves per forward pass.

        Each batch collects `batch_size` leaves via virtual-loss selection, sends
        them all to the neural network at once, then backpropagates results in
        parallel using a thread pool.
        """
        assert self._root is not None
        assert self.batch_policy_value_fn is not None

        remaining = self.simulations
        while remaining > 0:
            batch_size = min(self.batch_size, remaining)
            remaining -= batch_size

            leaves_and_legals = self._collect_leaves_with_virtual_loss(batch_size)
            batch_results = self.batch_policy_value_fn(
                [(node.game, legal) for node, legal in leaves_and_legals]
            )
            self._backpropagate_batch(leaves_and_legals, batch_results)

    def _collect_leaves_with_virtual_loss(
        self, count: int
    ) -> list[tuple[AZNode, list[Move]]]:
        """Select `count` leaf nodes for evaluation, applying virtual loss to each.

        Virtual loss makes each selected node look artificially bad to subsequent
        selections in the same batch, encouraging the batch to explore different paths
        rather than all converging on the same node before any backprop has occurred.
        """
        assert self._root is not None
        leaves_and_legals: list[tuple[AZNode, list[Move]]] = []
        for _ in range(count):
            with self._lock:
                node = self._select_with_virtual_loss(self._root)
                if node.is_terminal or node.is_round_boundary:
                    legal: list[Move] = []
                else:
                    legal = self._canonical_moves(node.game)
                leaves_and_legals.append((node, legal))
        return leaves_and_legals

    def _backpropagate_batch(
        self,
        leaves_and_legals: list[tuple[AZNode, list[Move]]],
        batch_results: list[tuple[list[float], float]],
    ) -> None:
        """Undo virtual losses and backpropagate values for an evaluated batch."""

        def _process_one_leaf(
            node: AZNode,
            legal: list[Move],
            priors: list[float],
            value: float,
        ) -> None:
            with self._lock:
                self._undo_virtual_loss(node)
                if node.net_value_diff is None and not node.is_terminal:
                    node.net_value_diff = value
                if node.is_terminal:
                    value = self._terminal_value(node.game)
                elif node._untried_moves is None:
                    node._untried_moves = list(legal)
                    node._untried_priors = list(priors)
                self._backpropagate(node, value)

        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            futures = [
                pool.submit(_process_one_leaf, node, legal, priors, value)
                for (node, legal), (priors, value) in zip(
                    leaves_and_legals, batch_results
                )
            ]
            for future in futures:
                future.result()

    # ── Virtual loss helpers ───────────────────────────────────────────────

    def _select_with_virtual_loss(self, root: AZNode) -> AZNode:
        """Descend the tree to a leaf, applying virtual loss at every visited node.

        Virtual loss is incremented at each node on the selected path so that
        subsequent calls in the same batch will prefer different paths. The losses
        are undone during backpropagation via _undo_virtual_loss().

        This is the batched equivalent of _select() — same traversal logic, but
        with virtual loss bookkeeping added at each step.
        """
        node = root
        while not node.is_terminal and not node.is_round_boundary:
            if node._untried_moves is None:
                node.virtual_loss += 1
                return node

            if node._untried_moves:
                node = self._expand_one_child_with_virtual_loss(node)
                return node

            eligible = [child for child in node.children if not child._fully_explored]
            if not eligible:
                node._explored = True
                node.virtual_loss += 1
                return node

            node.virtual_loss += 1
            parent_visits_with_virtual = node.visits + node.virtual_loss
            assert node.net_value_diff is not None
            unvisited_exploitation = node.net_value_diff
            node = max(
                eligible,
                key=lambda child: child.puct_score(
                    parent_visits_with_virtual, unvisited_exploitation
                ),
            )

        node.virtual_loss += 1
        return node

    def _expand_one_child_with_virtual_loss(self, parent: AZNode) -> AZNode:
        """Pop one untried move, create a child node, and apply virtual loss to both."""
        move = parent._untried_moves.pop()  # type: ignore[union-attr]
        prior = parent._untried_priors.pop()  # type: ignore[union-attr]
        child_game = parent.game.clone()
        child_game.make_move(move)
        child_game.next_player()
        child = self._make_node(child_game, parent=parent, move=move, prior=prior)
        parent.children.append(child)
        parent.virtual_loss += 1
        child.virtual_loss += 1
        return child

    def _undo_virtual_loss(self, node: AZNode) -> None:
        """Walk up from node to root, decrementing virtual loss at each ancestor.

        Every node on the path selected by _select_with_virtual_loss() had its
        virtual_loss incremented by 1. This method reverses those increments so
        that real backpropagated values are not contaminated by the temporary penalty.
        """
        current: AZNode | None = node
        while current is not None:
            if current.virtual_loss > 0:
                current.virtual_loss -= 1
            current = current.parent

    # ── Tree operations ────────────────────────────────────────────────────

    def _select(self, node: AZNode) -> AZNode:
        """Descend the tree to a leaf node for evaluation (single-threaded path).

        At each level:
          - If the node has never been evaluated (untried_moves is None), return it.
          - If the node has untried moves, expand one and return the new child.
          - Otherwise, follow the child with the highest PUCT score.

        Returns a terminal, round-boundary, or newly-expanded leaf node.
        """
        while not node.is_terminal and not node.is_round_boundary:
            if node._untried_moves is None:
                return node

            if node._untried_moves:
                node = self._expand_one_child(node)
                return node

            eligible = [child for child in node.children if not child._fully_explored]
            if not eligible:
                node._explored = True
                return node
            assert node.net_value_diff is not None
            unvisited_exploitation = node.net_value_diff
            node = max(
                eligible,
                key=lambda child: child.puct_score(node.visits, unvisited_exploitation),
            )

        return node

    def _expand_one_child(self, parent: AZNode) -> AZNode:
        """Pop one untried move and create a child node (single-threaded path)."""
        move = parent._untried_moves.pop()  # type: ignore[union-attr]
        prior = parent._untried_priors.pop()  # type: ignore[union-attr]
        child_game = parent.game.clone()
        child_game.make_move(move)
        child_game.next_player()
        child = self._make_node(child_game, parent=parent, move=move, prior=prior)
        parent.children.append(child)
        return child

    def _ensure_expanded(self, node: AZNode) -> float | None:
        """Evaluate node with the policy network if it has not been visited yet.

        Sets _untried_moves and _untried_priors from the policy output. Returns
        the value estimate from the network, or None if the node was already
        expanded or has no legal moves.
        """
        if node._untried_moves is not None:
            return None
        if node.is_terminal or node.is_round_boundary:
            node._untried_moves = []
            node._untried_priors = []
            return None
        legal = self._canonical_moves(node.game)
        if not legal:
            node._untried_moves = []
            node._untried_priors = []
            return None
        priors, value = self.policy_value_fn(node.game, legal)
        node._untried_moves = list(legal)
        node._untried_priors = list(priors)
        if node.net_value_diff is None:
            node.net_value_diff = value
        return value

    def _evaluate(self, node: AZNode) -> float:
        """Return a value estimate for a leaf node.

        Evaluation priority:
          1. Terminal node: use exact score differential.
          2. Round boundary: use heuristic or neural net value.
          3. Unvisited node: call policy_value_fn to expand and get value.
          4. Already expanded: call policy_value_fn again (this node was selected
             but its prior expansion produced no useful value, e.g. no legal moves).
        """
        if node.is_terminal:
            return self._terminal_value(node.game)

        if node.is_round_boundary:
            return self._evaluate_round_boundary(node)

        if node._untried_moves is None:
            expansion_value = self._ensure_expanded(node)
            if expansion_value is not None:
                return expansion_value

        # Node was already expanded but we need a fresh value estimate.
        _, value = self.policy_value_fn(node.game, self._canonical_moves(node.game))
        if node.net_value_diff is None:
            node.net_value_diff = value
        return value

    def _evaluate_round_boundary(self, node: AZNode) -> float:
        """Return a value estimate for a round-boundary node.

        If use_heuristic_value is True, computes value from the actual score
        differential rather than calling the neural net.
        """
        if self.use_heuristic_value:
            return self._terminal_value(node.game)
        _, value = self.policy_value_fn(node.game, [])
        if node.net_value_diff is None:
            node.net_value_diff = value
        return value

    def _backpropagate(self, node: AZNode, value: float) -> None:
        """Walk up from node to root, updating visit counts and total values.

        The value is negated at each level because the parent node is always
        the opponent's turn: a value of +1.0 for the child (good for child's player)
        is -1.0 for the parent (bad for parent's player).
        """
        current: AZNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            value = -value
            current._check_and_mark_explored()
            current = current.parent

    # ── Move selection ─────────────────────────────────────────────────────

    def _pick_move(self, root: AZNode) -> Move:
        """Select the move to play from the root after simulations are complete.

        With temperature=0.0: returns the move with the most visits (deterministic).
        With temperature>0.0: samples proportional to visits^(1/temperature) using
        the 25-move temperature rule — high temperature for the first 25 moves,
        low temperature after.

        Falls back to the first legal move if no children were created (e.g.
        simulations=0 or the root was immediately terminal).
        """
        if not root.children:
            legal = root.game.legal_moves()
            if not legal:
                raise RuntimeError(
                    "_pick_move called on a terminal/boundary node with no children"
                )
            return legal[0]

        if self.temperature == 0.0:
            best = max(root.children, key=lambda child: child.visits)
            assert best.move is not None
            return best.move

        return self._sample_move_by_temperature(root)

    def _sample_move_by_temperature(self, root: AZNode) -> Move:
        """Sample a move proportional to visit counts raised to 1/temperature.

        Uses the 25-move temperature rule: high temperature (exploratory) for
        the first EXPLORATION_MOVES moves, low temperature (decisive) after.
        """
        move_count = root.game.turn
        temperature = (
            EXPLORATION_TEMP if move_count < EXPLORATION_MOVES else DETERMINISTIC_TEMP
        )
        visit_weights = torch.tensor(
            [child.visits ** (1.0 / temperature) for child in root.children],
            dtype=torch.float32,
        )
        probabilities = visit_weights / visit_weights.sum()
        chosen_index = int(torch.multinomial(probabilities, num_samples=1).item())
        chosen_move = root.children[chosen_index].move
        assert chosen_move is not None
        return chosen_move

    # ── Value computation ──────────────────────────────────────────────────

    def _terminal_value(self, game: Game) -> float:
        """Compute score-differential value at a terminal or round-boundary node.

        Returns a value in [-1.0, 1.0] from the current player's perspective.
        Positive means the current player is ahead.
        """
        current_player_index = game.current_player_index
        my_earned = game.players[current_player_index].earned
        opponent_earned = game.players[1 - current_player_index].earned
        diff = (my_earned - opponent_earned) / _SCORE_DIFF_DIVISOR
        return max(-1.0, min(1.0, diff))

    def _minimax_value(
        self, node: AZNode, depth: int = 0, root_player: int | None = None
    ) -> float:
        """Compute a minimax rollup of backed-up values through the search tree.

        Used by the inspector UI to display a cumulative value estimate that
        accounts for both players playing optimally from each position.

        All values are returned from root_player's perspective.
        """
        if root_player is None:
            root_player = node.game.current_player_index

        if not node.children or node.is_round_boundary or node.is_terminal:
            return self._leaf_value_from_root_perspective(node, root_player)

        visited_children = [child for child in node.children if child.visits > 0]
        if not visited_children:
            return self._node_value_from_root_perspective(node, root_player)

        if node.game.current_player_index == root_player:
            return max(
                self._minimax_value(child, depth + 1, root_player)
                for child in visited_children
            )
        else:
            return min(
                self._minimax_value(child, depth + 1, root_player)
                for child in visited_children
            )

    def _leaf_value_from_root_perspective(
        self, node: AZNode, root_player: int
    ) -> float:
        """Return this leaf node's q_value (or heuristic) from root_player's
        perspective."""
        leaf_value = (
            node.q_value if node.visits > 0 else self._terminal_value(node.game)
        )
        if node.game.current_player_index == root_player:
            return leaf_value
        return -leaf_value

    def _node_value_from_root_perspective(
        self, node: AZNode, root_player: int
    ) -> float:
        """Return this node's q_value from root_player's perspective."""
        if node.game.current_player_index == root_player:
            return node.q_value
        return -node.q_value

    # ── Stability tracking ─────────────────────────────────────────────────

    def is_stable(self) -> bool:
        """Return True if the search tree is fully explored and further simulations
        would have no effect."""
        if self._root is None:
            return True
        return self._root._fully_explored

    def record_batch_stability(self, top_k: int = 5) -> None:
        """Track whether the top-k root children by visit count have stabilized.

        Increments _stable_batches when the ranked list of top-k children is
        unchanged from the previous call. Resets to zero when the ranking changes.
        Used by the inspector UI to decide when to stop showing a spinner.
        """
        if self._root is not None:
            self._is_subtree_explored(self._root)

        if self._root is None or not self._root.children:
            self._stable_batches = 0
            self._last_top_k = []
            return

        visited = [child for child in self._root.children if child.visits > 0]
        current_top_k = [
            id(child)
            for child in sorted(visited, key=lambda child: child.visits, reverse=True)[
                :top_k
            ]
        ]

        if current_top_k == self._last_top_k:
            self._stable_batches += 1
        else:
            self._stable_batches = 0
        self._last_top_k = current_top_k

    def _is_subtree_explored(self, node: AZNode) -> bool:
        """Recursively check and mark whether the subtree below node is fully explored.

        This is a post-search pass that ensures _explored flags are set correctly
        after batched search (which may not mark every node during backprop).
        """
        if node._explored:
            return True
        if node.is_terminal or node.is_round_boundary:
            return True
        if not node.is_fully_expanded:
            return False
        if not node.children:
            node._explored = True
            return True
        if all(self._is_subtree_explored(child) for child in node.children):
            node._explored = True
            return True
        return False

    # ── Serialization (inspector UI) ───────────────────────────────────────

    def serialize(self, max_depth: int = 4, top_k: int = 200) -> dict:
        """Serialize the search tree to a JSON-compatible dict for the inspector UI.

        Args:
            max_depth: Maximum depth of nodes to include in the output.
            top_k:     Maximum number of children to include per node (by visit count).
        """
        if self._root is None:
            return self._empty_node_dict()
        root_player = self._root.game.current_player_index
        return self._serialize_node(
            self._root,
            depth=0,
            max_depth=max_depth,
            top_k=top_k,
            root_player=root_player,
        )

    def _empty_node_dict(self) -> dict:
        """Return a placeholder node dict for when the tree has no root."""
        return {
            "key": "0",
            "move": None,
            "visits": 0,
            "value_diff": 0.0,
            "prior": 0.0,
            "immediate": None,
            "cumulative_immediate": None,
            "minimax_value": 0.0,
            "net_value_diff": None,
            "visit_fraction": None,
            "is_round_boundary": False,
            "depth": 0,
            "children": [],
        }

    def _serialize_node(
        self,
        node: AZNode,
        depth: int,
        max_depth: int,
        top_k: int,
        root_player: int | None = None,
        parent_game: Game | None = None,
        parent_visits: int | None = None,
    ) -> dict:
        """Serialize one node and its visible children to a dict.

        Delegates to helpers for immediate score, children, and cumulative score
        to keep each responsibility separate.
        """
        if root_player is None:
            root_player = node.game.current_player_index
        if parent_game is None:
            parent_game = node.game

        immediate = self._compute_immediate_score(node, parent_game, root_player)
        serialized_children = self._serialize_children(
            node, depth, max_depth, top_k, root_player
        )
        cumulative_immediate = self._compute_cumulative_score(
            node, immediate, serialized_children, root_player
        )

        return {
            "key": hex(node.zobrist_hash),
            "move": _move_str(node.move) if node.move is not None else None,
            "visits": node.visits,
            "immediate": immediate,
            "cumulative_immediate": cumulative_immediate,
            "value_diff": self._q_value_from_root_perspective(node, root_player),
            "minimax_value": float(self._minimax_value(node, depth, root_player)),
            "prior": float(node.prior),
            "is_round_boundary": bool(node.is_round_boundary),
            "depth": depth,
            "children": serialized_children,
            "net_value_diff": self._net_value_from_root_perspective(node, root_player),
            "visit_fraction": (
                node.visits / parent_visits
                if parent_visits is not None and parent_visits > 0
                else None
            ),
        }

    def _compute_immediate_score(
        self, node: AZNode, parent_game: Game, root_player: int
    ) -> float | None:
        """Compute the immediate score delta from the move that reached this node.

        Returns the change in the moving player's `earned` score caused by their
        move, from the root player's perspective (negated if the opponent moved).
        Returns None for the root node (no move was made to reach it).
        """
        if node.move is None:
            return None
        moving_player_index = parent_game.current_player_index
        score_before = parent_game.players[moving_player_index].earned
        score_after = node.game.players[moving_player_index].earned
        delta = score_after - score_before
        return delta if moving_player_index == root_player else -delta

    def _serialize_children(
        self,
        node: AZNode,
        depth: int,
        max_depth: int,
        top_k: int,
        root_player: int,
    ) -> list[dict]:
        """Serialize the top-k visited children of node, deduplicated by Zobrist hash.

        Children are sorted by cumulative_immediate score, alternating descending
        (root player's turn) and ascending (opponent's turn) by depth.
        """
        if depth >= max_depth or not node.children:
            return []

        visited_children = [child for child in node.children if child.visits > 0]
        top_children = sorted(
            visited_children, key=lambda child: child.visits, reverse=True
        )[:top_k]
        deduped_children = self._deduplicate_by_hash(top_children)

        serialized = [
            self._serialize_node(
                child,
                depth + 1,
                max_depth,
                top_k,
                root_player,
                node.game,
                parent_visits=node.visits,
            )
            for child in deduped_children
        ]

        serialized.sort(
            key=lambda child_dict: (
                child_dict["cumulative_immediate"]
                if child_dict["cumulative_immediate"] is not None
                else float("-inf")
            ),
            reverse=(depth % 2 == 0),
        )
        return serialized

    def _deduplicate_by_hash(self, nodes: list[AZNode]) -> list[AZNode]:
        """Remove duplicate nodes that share the same Zobrist hash (transpositions)."""
        seen_hashes: set[int] = set()
        deduped: list[AZNode] = []
        for node in nodes:
            if node.zobrist_hash not in seen_hashes:
                seen_hashes.add(node.zobrist_hash)
                deduped.append(node)
        return deduped

    def _compute_cumulative_score(
        self,
        node: AZNode,
        immediate: float | None,
        serialized_children: list[dict],
        root_player: int,
    ) -> float | None:
        """Compute the minimax rollup of immediate scores from this node down.

        This is the sum of immediate score deltas along the best (or worst, for
        the opponent) path through the already-serialized subtree. It gives the
        inspector UI a single number representing "what's the best outcome from here."

        Returns None for the root node (no immediate score to anchor the rollup).
        """
        if immediate is None:
            return None
        if not serialized_children:
            return immediate

        children_with_scores = [
            child
            for child in serialized_children
            if child["cumulative_immediate"] is not None
        ]
        if not children_with_scores:
            return immediate

        next_player_index = node.game.current_player_index
        if next_player_index == root_player:
            best_child_score = max(
                child["cumulative_immediate"] for child in children_with_scores
            )
        else:
            best_child_score = min(
                child["cumulative_immediate"] for child in children_with_scores
            )
        return immediate + best_child_score

    def _q_value_from_root_perspective(self, node: AZNode, root_player: int) -> float:
        """Return this node's q_value adjusted to be from root_player's perspective."""
        if node.game.current_player_index == root_player:
            return float(node.q_value)
        return float(-node.q_value)

    def _net_value_from_root_perspective(
        self, node: AZNode, root_player: int
    ) -> float | None:
        """Return net_value_diff adjusted to be from root_player's perspective, or
        None."""
        if node.net_value_diff is None:
            return None
        if node.game.current_player_index == root_player:
            return float(node.net_value_diff)
        return float(-node.net_value_diff)

    # ── Node factory and pruning ───────────────────────────────────────────

    def _make_node(
        self,
        game: Game,
        parent: AZNode | None,
        move: Move | None,
        prior: float,
    ) -> AZNode:
        """Create a new AZNode, register it in the transposition table, and return
        it."""
        zobrist_hash = _ZOBRIST.hash_state(game)
        node = AZNode(
            game=game,
            zobrist_hash=zobrist_hash,
            parent=parent,
            move=move,
            prior=prior,
        )
        self._table[zobrist_hash] = node
        return node

    def _prune(self, node: AZNode) -> None:
        """Recursively remove node and all descendants from the transposition table."""
        self._table.pop(node.zobrist_hash, None)
        for child in node.children:
            self._prune(child)

    def _canonical_moves(self, game: Game) -> list[Move]:
        """Return legal moves with duplicate factory sources deduplicated.

        Two factories with identical tile multisets are interchangeable. Keeping
        only moves from one of each pair reduces the branching factor without
        losing any strategically distinct options.
        """
        legal = game.legal_moves()
        if not legal:
            return legal

        canonical_factory_index = self._build_canonical_factory_map(game)

        result = []
        for move in legal:
            if move.source == CENTER:
                result.append(move)
            elif canonical_factory_index.get(move.source, move.source) == move.source:
                result.append(move)
        return result

    def _build_canonical_factory_map(self, game: Game) -> dict[int, int]:
        """Map each factory index to the lowest-index factory with identical contents.

        Factories with the same multiset of tiles are interchangeable, so we keep
        only the first occurrence of each unique factory composition.
        """
        first_seen: dict[tuple, int] = {}
        canonical_index: dict[int, int] = {}
        for factory_index, factory in enumerate(game.factories):
            contents_key = tuple(sorted(tile.name for tile in factory))
            if contents_key in first_seen:
                canonical_index[factory_index] = first_seen[contents_key]
            else:
                first_seen[contents_key] = factory_index
                canonical_index[factory_index] = factory_index
        return canonical_index


# ── Serialization helpers ──────────────────────────────────────────────────


def _move_str(move: Move) -> str:
    """Format a move as a human-readable string for the inspector UI.

    Example outputs: "CTR Blue → row 2", "F3 Red → floor"
    """
    source = "CTR" if move.source == CENTER else f"F{move.source + 1}"
    color = move.tile.name.capitalize()
    destination = (
        "floor" if move.destination == FLOOR else f"row {move.destination + 1}"
    )
    return f"{source} {color} → {destination}"
