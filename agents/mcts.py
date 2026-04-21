# agents/mcts.py

"""Monte Carlo Tree Search agent for Azul."""

import copy
import math
import logging

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

    Each node represents a game state reached by a particular move.
    The root node has move=None and parent=None.

    Attributes:
        game:          A complete copy of the game state at this node.
        move:          The move that was applied to reach this node (None for root).
        parent:        The parent MCTSNode (None for root).
        children:      MCTSNodes that have been expanded from this node.
        untried_moves: Legal moves not yet expanded into children.
        visits:        Number of times this node has been visited.
        total_value:   Sum of simulation outcomes backpropagated through this node.
    """

    def __init__(self, *, game: Game, move: Move | None, parent: "MCTSNode | None"):
        self.game = game
        self.move = move
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.untried_moves: list[Move] = game.legal_moves()
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

        The game passed in is never modified — we work on deep copies internally.
        """
        root = MCTSNode(game=copy.deepcopy(game), move=None, parent=None)

        for _ in range(self.simulations):
            node = self._select(root)
            node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # After all simulations, pick the most-visited child of the root.
        # Most-visited (not highest UCB1) is the standard choice at decision
        # time — it's more robust to outliers than picking by win rate alone.
        best = max(root.children, key=lambda n: n.visits)
        assert best.move is not None  # root's children always have moves
        return best.move

    # ── The four MCTS steps ───────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk down the tree, choosing the best child at each level.

        Stop when we reach a node that still has untried moves (we'll expand
        it next) or a terminal node (game over).
        """
        while not node.game.is_game_over():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add one new child by trying an unexplored move.

        If the node is terminal (no untried moves because the game is over,
        not because all moves are expanded), return it as-is.
        """
        if not node.untried_moves:
            return node

        move = node.untried_moves.pop()
        new_game = copy.deepcopy(node.game)
        new_game.make_move(move)
        new_game.advance()
        child = MCTSNode(game=new_game, move=move, parent=node)
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """Play out the game randomly from this node and return the result.

        Result is from the perspective of the root player (player 0):
            1.0 = player 0 wins
            0.0 = player 0 loses
            0.5 = tie

        We work on a copy so the node's game state is never touched.
        """
        sim_game = copy.deepcopy(node.game)
        sim_game.advance()  # guard: node may be between rounds
        random_agent = _RandomRolloutAgent()

        while not sim_game.is_game_over():
            move = random_agent.choose_move(sim_game)
            sim_game.make_move(move)
            sim_game.advance()

        scores = [p.score for p in sim_game.state.players]
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]

        if len(winners) > 1:
            return 0.5  # tie
        return 1.0 if winners[0] == 0 else 0.0

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Walk back up to the root, adding the result to every ancestor."""
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += result
            current = current.parent


# ── Internal rollout helper ───────────────────────────────────────────────────


class _RandomRolloutAgent:
    """Minimal random agent used only for rollout simulations.

    Kept private to this module — it's not part of the public agent interface.
    We don't use RandomAgent here because we want the simplest possible rollout:
    pick any legal move uniformly. RandomAgent's heuristics add overhead and
    could subtly bias value estimates.
    """

    def choose_move(self, game: Game) -> Move:
        import random

        return random.choice(game.legal_moves())
