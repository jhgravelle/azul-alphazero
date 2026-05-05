# agents/mcts.py

"""Monte Carlo Tree Search agent for Azul."""

import math
import logging
import random

from agents.base import Agent
from engine.game import Game, Move

logger = logging.getLogger(__name__)


def ucb1(
    *, visits: int, total_value: float, parent_visits: int, c: float = math.sqrt(2)
) -> float:
    """Return the UCB1 score for a node.

    Args:
        visits:        How many times this node has been visited.
        total_value:   Sum of all simulation results backpropagated through it.
        parent_visits: How many times the parent node has been visited.
        c:             Exploration constant. Higher = more exploration.
                       sqrt(2) is the theoretically motivated default.

    Returns:
        math.inf for unvisited nodes (they must be tried before any visited node).
        Otherwise: (total_value / visits) + c * sqrt(ln(parent_visits) / visits)
    """
    if visits == 0:
        return math.inf
    exploitation = total_value / visits
    exploration = c * math.sqrt(math.log(parent_visits) / visits)
    return exploitation + exploration


class MCTSNode:
    """A single node in the Monte Carlo search tree.

    Nodes do NOT store a game copy. The game state at any node is
    reconstructed by replaying moves from the root on demand. This avoids
    a deepcopy per simulation — the single most expensive operation in the
    naive implementation.

    Attributes:
        move:          The move that was applied to reach this node (None for root).
        parent:        The parent MCTSNode (None for root).
        children:      MCTSNodes expanded from this node.
        untried_moves: Legal moves not yet expanded into children.
        visits:        Number of times this node has been visited.
        total_value:   Sum of simulation outcomes backpropagated through this node.
    """

    def __init__(
        self, *, move: Move | None, parent: "MCTSNode | None", untried_moves: list[Move]
    ):
        self.move = move
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.untried_moves: list[Move] = untried_moves
        self.visits: int = 0
        self.total_value: float = 0.0

    def is_fully_expanded(self) -> bool:
        """Return True when every legal move has been expanded into a child."""
        return len(self.untried_moves) == 0

    def best_child(self, c: float = math.sqrt(2)) -> "MCTSNode":
        """Return the child with the highest UCB1 score."""
        return max(
            self.children,
            key=lambda child: ucb1(
                visits=child.visits,
                total_value=child.total_value,
                parent_visits=self.visits,
                c=c,
            ),
        )

    def path_from_root(self) -> list[Move]:
        """Return the sequence of moves from root to this node."""
        moves = []
        current: MCTSNode | None = self
        while current is not None and current.move is not None:
            moves.append(current.move)
            current = current.parent
        moves.reverse()
        return moves


class MCTSAgent(Agent):
    """An agent that uses Monte Carlo Tree Search to choose moves.

    Args:
        simulations: Number of simulations (iterations) to run per move.
                     Higher = stronger but slower. 200 is a reasonable default.
    """

    def __init__(self, simulations: int = 200):
        self.simulations = simulations

    def choose_move(self, game: Game) -> Move:
        """Run MCTS from the current game state and return the best move found.

        The game passed in is never modified — we clone it once at the start
        and replay moves onto working copies per simulation.
        """
        root_game = game.clone()
        root = MCTSNode(move=None, parent=None, untried_moves=root_game.legal_moves())

        for _ in range(self.simulations):
            node, sim_game = self._select(root, root_game)
            node, sim_game = self._expand(node, sim_game)
            result = self._simulate(sim_game)
            self._backpropagate(node, result)

        best = max(root.children, key=lambda n: n.visits)
        assert best.move is not None
        return best.move

    # ── The four MCTS steps ───────────────────────────────────────────────

    def _select(self, node: MCTSNode, root_game: Game) -> tuple[MCTSNode, Game]:
        """Walk down the tree to a node with untried moves (or a terminal).

        Returns the selected node and a game copy advanced to that node's state.
        The game is reconstructed by replaying the path from root — one clone,
        then O(depth) make_move calls, no further copies.
        """
        while not node.untried_moves and node.children:
            node = node.best_child()

        sim_game = self._replay_to_node(node, root_game)
        return node, sim_game

    def _expand(self, node: MCTSNode, sim_game: Game) -> tuple[MCTSNode, Game]:
        """Add one new child by trying an unexplored move.

        Returns the new child node and the game advanced to the child's state.
        No additional clone needed — we continue mutating sim_game in place.
        """
        if not node.untried_moves:
            return node, sim_game

        move = node.untried_moves.pop()
        sim_game.make_move(move)
        sim_game.advance()
        child = MCTSNode(
            move=move,
            parent=node,
            untried_moves=sim_game.legal_moves(),
        )
        node.children.append(child)
        return child, sim_game

    def _simulate(self, sim_game: Game) -> float:
        """Play out the current round randomly and return a heuristic result.

        Stops at the round boundary rather than playing to game over. This
        cuts rollout length from ~150 moves to ~10-20, making each simulation
        much cheaper without meaningfully weakening move selection.

        Result is from the perspective of player 0:
            1.0 = player 0 winning, 0.0 = losing, 0.5 = tied
        """
        while sim_game.legal_moves():
            move = random.choice(sim_game.legal_moves())
            sim_game.make_move(move)
            round_ended = sim_game.advance()
            if round_ended:
                break

        scores = [p.earned for p in sim_game.players]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        if len(winners) > 1:
            return 0.5
        return 1.0 if winners[0] == 0 else 0.0

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Walk back up to the root, adding the result to every ancestor."""
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += result
            current = current.parent

    # ── Internal helpers ──────────────────────────────────────────────────

    def _replay_to_node(self, node: MCTSNode, root_game: Game) -> Game:
        """Clone root_game and replay moves to reach node's game state."""
        sim_game = root_game.clone()
        for move in node.path_from_root():
            sim_game.make_move(move)
            sim_game.advance()
        return sim_game
