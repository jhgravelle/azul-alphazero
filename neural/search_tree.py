# neural/search_tree.py
"""Game-owned MCTS search tree with transposition table and subtree reuse."""

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

_ZOBRIST = ZobristTable()


def make_policy_value_fn(
    net: "AzulNet",
    device: "torch.device | None" = None,
) -> "PolicyValueFn":
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
            logits, _value_win, value_diff, _value_abs = net(spatial, flat)
        value = value_diff
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

    _untried_moves: list[Move] | None = None
    _untried_priors: list[float] | None = None
    _explored: bool = False
    net_value_diff: float | None = None

    @property
    def q_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    @property
    def is_terminal(self) -> bool:
        return self.game.is_game_over()

    @property
    def is_round_boundary(self) -> bool:
        """True when all tiles have been taken but the round is not yet scored."""
        return self.game.is_round_over() and not self.game.is_game_over()

    @property
    def is_fully_expanded(self) -> bool:
        return self._untried_moves is not None and len(self._untried_moves) == 0

    @property
    def _fully_explored(self) -> bool:
        if self.is_terminal or self.is_round_boundary:
            return True
        return self._explored

    def _check_and_mark_explored(self) -> bool:
        if self._explored:
            return True
        if self.is_terminal or self.is_round_boundary:
            self._explored = True
            return True
        if not self.is_fully_expanded:
            return False
        if not self.children:
            self._explored = True
            return True
        if all(c._fully_explored for c in self.children):
            self._explored = True
            return True
        return False

    def puct_score(self, parent_visits: int) -> float:
        if self._fully_explored:
            return float("-inf")
        adj_visits = self.visits + self.virtual_loss
        if adj_visits == 0:
            q = 0.0
        else:
            q = -(self.total_value + float(self.virtual_loss)) / adj_visits
        u = _PUCT_C * self.prior * math.sqrt(parent_visits) / (1 + adj_visits)
        return q + u


# ── Policy/value callable types ───────────────────────────────────────────────

PolicyValueFn = Callable[[Game, list[Move]], tuple[list[float], float]]
BatchPolicyValueFn = Callable[
    [list[tuple[Game, list[Move]]]],
    list[tuple[list[float], float]],
]


def make_batch_policy_value_fn(
    net: "AzulNet",
    device: "torch.device | None" = None,
) -> "BatchPolicyValueFn":
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

        spatial_t = torch.stack(spatials).to(device)
        flat_t = torch.stack(flats).to(device)

        net.eval()
        with torch.no_grad():
            logits_b, _wins_b, values_b, _abs_b = net(spatial_t, flat_t)

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
        self._table = {}
        self._root = self._make_node(game.clone(), parent=None, move=None, prior=0.0)
        self._stable_batches = 0
        self._last_top_k = []

    def choose_move(self, game: Game) -> Move:
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

        if (
            new_root is not None
            and new_root.children == []
            and new_root._untried_moves == []
        ):
            new_root = None

        if new_root is None:
            new_game = self._root.game.clone()
            new_game.make_move(move)
            new_game.advance()
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
        if self._root is not None and self._root._fully_explored:
            return
        if self.batch_policy_value_fn is not None:
            self._run_batched_simulations()
        else:
            assert self._root is not None
            for _ in range(self.simulations):
                if self._root._fully_explored:
                    break
                node = self._select(self._root)
                value = self._evaluate(node)
                self._backpropagate(node, value)

    def _run_batched_simulations(self) -> None:
        assert self._root is not None
        assert self.batch_policy_value_fn is not None

        remaining = self.simulations
        while remaining > 0:
            batch_n = min(self.batch_size, remaining)
            remaining -= batch_n

            leaves_and_legals: list[tuple[AZNode, list[Move]]] = []
            for _ in range(batch_n):
                with self._lock:
                    node = self._select_vl(self._root)
                    if node.is_terminal or node.is_round_boundary:
                        legal: list[Move] = []
                    else:
                        legal = self._canonical_moves(node.game)
                    leaves_and_legals.append((node, legal))

            batch_results = self.batch_policy_value_fn(
                [(node.game, legal) for node, legal in leaves_and_legals]
            )

            def _backprop_one(
                node: AZNode,
                legal: list[Move],
                priors: list[float],
                value: float,
            ) -> None:
                with self._lock:
                    self._undo_vl(node)
                    if node.net_value_diff is None and not node.is_terminal:
                        node.net_value_diff = value
                    if node.is_terminal:
                        value = self._terminal_value(node.game)
                    elif node._untried_moves is None:
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
                    f.result()

    # ── Virtual loss helpers ───────────────────────────────────────────────

    def _select_vl(self, root: AZNode) -> AZNode:
        node = root
        while not node.is_terminal and not node.is_round_boundary:
            if node._untried_moves is None:
                node.virtual_loss += 1
                return node
            if node._untried_moves:
                move = node._untried_moves.pop()
                prior = node._untried_priors.pop()  # type: ignore[union-attr]
                child_game = node.game.clone()
                child_game.make_move(move)
                child_game.next_player()
                child = self._make_node(child_game, parent=node, move=move, prior=prior)
                node.children.append(child)
                node.virtual_loss += 1
                child.virtual_loss += 1
                return child
            eligible = [c for c in node.children if not c._fully_explored]
            if not eligible:
                node._explored = True
                node.virtual_loss += 1
                return node
            node.virtual_loss += 1
            parent_visits = node.visits + node.virtual_loss
            node = max(eligible, key=lambda c: c.puct_score(parent_visits))
        node.virtual_loss += 1
        return node

    def _undo_vl(self, node: AZNode) -> None:
        current: AZNode | None = node
        while current is not None:
            if current.virtual_loss > 0:
                current.virtual_loss -= 1
            current = current.parent

    # ── Tree operations ────────────────────────────────────────────────────

    def _select(self, node: AZNode) -> AZNode:
        while not node.is_terminal and not node.is_round_boundary:
            if node._untried_moves is None:
                return node
            if node._untried_moves:
                move = node._untried_moves.pop()
                prior = node._untried_priors.pop()  # type: ignore[union-attr]
                child_game = node.game.clone()
                child_game.make_move(move)
                child_game.next_player()
                child = self._make_node(child_game, parent=node, move=move, prior=prior)
                node.children.append(child)
                return child
            eligible = [c for c in node.children if not c._fully_explored]
            if not eligible:
                node._explored = True
                return node
            node = max(eligible, key=lambda c: c.puct_score(node.visits))
        return node

    def _ensure_expanded(self, node: AZNode) -> float | None:
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
        if node.is_terminal:
            return self._terminal_value(node.game)
        if node.is_round_boundary:
            if self.use_heuristic_value:
                return self._terminal_value(node.game)
            _, value = self.policy_value_fn(node.game, [])
            if node.net_value_diff is None:
                node.net_value_diff = value
            return value
        if node._untried_moves is None:
            value = self._ensure_expanded(node)
            if value is not None:
                return value
        _, value = self.policy_value_fn(node.game, self._canonical_moves(node.game))
        if node.net_value_diff is None:
            node.net_value_diff = value
        return value

    def _backpropagate(self, node: AZNode, value: float) -> None:
        current: AZNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            value = -value
            c = current
            c._check_and_mark_explored()
            current = current.parent

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_node(
        self,
        game: Game,
        parent: AZNode | None,
        move: Move | None,
        prior: float,
    ) -> AZNode:
        h = _ZOBRIST.hash_state(game)
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
        self._table.pop(node.zobrist_hash, None)
        for child in node.children:
            self._prune(child)

    def _canonical_moves(self, game: Game) -> list[Move]:
        legal = game.legal_moves()
        if not legal:
            return legal

        seen: dict[tuple, int] = {}
        canonical: dict[int, int] = {}
        for f_idx, factory in enumerate(game.factories):
            key = tuple(sorted(t.name for t in factory))
            if key in seen:
                canonical[f_idx] = seen[key]
            else:
                seen[key] = f_idx
                canonical[f_idx] = f_idx

        result = []
        for move in legal:
            if move.source == CENTER:
                result.append(move)
            elif canonical.get(move.source, move.source) == move.source:
                result.append(move)
        return result

    def _pick_move(self, root: AZNode) -> Move:
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
        idx = game.current_player_index
        my_earned = game.players[idx].earned
        opp_earned = game.players[1 - idx].earned
        diff = (my_earned - opp_earned) / 20.0
        return max(-1.0, min(1.0, diff))

    def _minimax_value(
        self, node: AZNode, depth: int = 0, root_player: int | None = None
    ) -> float:
        if root_player is None:
            root_player = node.game.current_player_index

        if not node.children or node.is_round_boundary or node.is_terminal:
            leaf_val = (
                node.q_value if node.visits > 0 else self._terminal_value(node.game)
            )
            leaf_player = node.game.current_player_index
            return leaf_val if leaf_player == root_player else -leaf_val

        visited = [c for c in node.children if c.visits > 0]
        if not visited:
            leaf_player = node.game.current_player_index
            return node.q_value if leaf_player == root_player else -node.q_value

        if node.game.current_player_index == root_player:
            return max(self._minimax_value(c, depth + 1, root_player) for c in visited)
        else:
            return min(self._minimax_value(c, depth + 1, root_player) for c in visited)

    def serialize(self, max_depth: int = 4, top_k: int = 200) -> dict:
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

    def is_stable(self) -> bool:
        if self._root is None:
            return True
        return self._root._fully_explored

    def record_batch_stability(self, top_k: int = 5) -> None:
        if self._root is not None:
            self._is_subtree_explored(self._root)

        if self._root is None or not self._root.children:
            self._stable_batches = 0
            self._last_top_k = []
            return

        visited = [c for c in self._root.children if c.visits > 0]
        current = [
            id(c) for c in sorted(visited, key=lambda c: c.visits, reverse=True)[:top_k]
        ]

        if current == getattr(self, "_last_top_k", []):
            self._stable_batches += 1
        else:
            self._stable_batches = 0
        self._last_top_k = current

    def _is_subtree_explored(self, node: AZNode) -> bool:
        if node._explored:
            return True
        if node.is_terminal or node.is_round_boundary:
            return True
        if not node.is_fully_expanded:
            return False
        if not node.children:
            node._explored = True
            return True
        if all(self._is_subtree_explored(c) for c in node.children):
            node._explored = True
            return True
        return False

    def _empty_node_dict(self) -> dict:
        return {
            "key": "0",
            "move": None,
            "visits": 0,
            "value_diff": 0.0,
            "prior": 0.0,
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
        if root_player is None:
            root_player = node.game.current_player_index
        if parent_game is None:
            parent_game = node.game

        immediate = None
        if node.move is not None:
            moving_player_index = parent_game.current_player_index
            before = parent_game.players[moving_player_index].earned
            after = node.game.players[moving_player_index].earned
            delta = after - before
            immediate = delta if moving_player_index == root_player else -delta

        children: list[dict] = []
        if depth < max_depth and node.children:
            visited_children = [c for c in node.children if c.visits > 0]
            top = sorted(visited_children, key=lambda c: c.visits, reverse=True)[:top_k]
            seen_keys: set[int] = set()
            deduped = []
            for c in top:
                if c.zobrist_hash not in seen_keys:
                    seen_keys.add(c.zobrist_hash)
                    deduped.append(c)
            children = [
                self._serialize_node(
                    c,
                    depth + 1,
                    max_depth,
                    top_k,
                    root_player,
                    node.game,
                    parent_visits=node.visits,
                )
                for c in deduped
            ]
            children.sort(
                key=lambda c: (
                    c["cumulative_immediate"]
                    if c["cumulative_immediate"] is not None
                    else float("-inf")
                ),
                reverse=(depth % 2 == 0),
            )

        if immediate is None:
            cumulative_immediate = None
        elif not children:
            cumulative_immediate = immediate
        else:
            valid_children = [
                c for c in children if c["cumulative_immediate"] is not None
            ]
            if not valid_children:
                cumulative_immediate = immediate
            else:
                next_player_index = node.game.current_player_index
                if next_player_index == root_player:
                    best = max(c["cumulative_immediate"] for c in valid_children)
                else:
                    best = min(c["cumulative_immediate"] for c in valid_children)
                cumulative_immediate = immediate + best

        return {
            "key": hex(node.zobrist_hash),
            "move": _move_str(node.move) if node.move is not None else None,
            "visits": node.visits,
            "immediate": immediate,
            "cumulative_immediate": cumulative_immediate,
            "value_diff": float(
                node.q_value
                if node.game.current_player_index == root_player
                else -node.q_value
            ),
            "minimax_value": float(self._minimax_value(node, depth, root_player)),
            "prior": float(node.prior),
            "is_round_boundary": bool(node.is_round_boundary),
            "depth": depth,
            "children": children,
            "net_value_diff": (
                float(
                    node.net_value_diff
                    if node.game.current_player_index == root_player
                    else -node.net_value_diff
                )
                if node.net_value_diff is not None
                else None
            ),
            "visit_fraction": (
                node.visits / parent_visits
                if parent_visits is not None and parent_visits > 0
                else None
            ),
        }


# ── Serialization helpers ──────────────────────────────────────────────────


def _move_str(move: Move) -> str:
    from engine.game import CENTER, FLOOR

    source = "CTR" if move.source == CENTER else f"F{move.source + 1}"
    color = move.tile.name.capitalize()
    dest = "floor" if move.destination == FLOOR else f"row {move.destination + 1}"
    return f"{source} {color} → {dest}"
