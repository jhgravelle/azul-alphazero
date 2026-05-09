# agents/mcts.py

"""Monte Carlo Tree Search agent for Azul.

This is the plain MCTS agent — no neural network. It uses random rollouts
(simulations) to estimate how good each move is, guided by UCB1 to balance
exploring new moves versus exploiting moves that have looked good so far.

Value convention: values are in [-1.0, 1.0] from the current node's player
perspective (+1.0 = current player wins, -1.0 = current player loses, 0.0 = tie).
Backpropagation negates the value at each level so that every node always
accumulates results from the perspective of the player whose turn it is at
that node. This matches the convention used by SearchTree.

This is a separate agent from AlphaZeroAgent, which uses SearchTree with a
neural network policy/value function and PUCT instead of UCB1.
"""

import math
import random

from agents.base import Agent
from engine.game import Game, Move


def ucb1(
    *, visits: int, total_value: float, parent_visits: int, c: float = math.sqrt(2)
) -> float:
    """Return the UCB1 score for a node.

    UCB1 balances exploitation (preferring moves with high average value) and
    exploration (preferring moves that haven't been tried much). Unvisited nodes
    always return infinity so they are tried before any visited node.

    Args:
        visits:        How many times this node has been visited.
        total_value:   Sum of all simulation results backpropagated through it.
        parent_visits: How many times the parent node has been visited.
        c:             Exploration constant. Higher = more exploration.
                       sqrt(2) is the theoretically motivated default.

    Returns:
        math.inf for unvisited nodes.
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

    Values are stored from the perspective of the player whose turn it is at
    this node (+1.0 = that player wins, -1.0 = that player loses, 0.0 = tie).
    Backpropagation negates the value at each level, so a win for the child's
    player becomes a loss from the parent's player's point of view.

    Attributes:
        move:          The move that was applied to reach this node (None for root).
        parent:        The parent MCTSNode (None for root).
        children:      MCTSNodes expanded from this node.
        untried_moves: Legal moves not yet expanded into children.
        visits:        Number of times this node has been visited.
        total_value:   Sum of simulation outcomes from the perspective of the player
                       whose turn it is at this node.
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

    def is_terminal(self) -> bool:
        """Return True when there are no untried moves and no children to explore."""
        return self.is_fully_expanded() and not self.children

    def best_child(self, exploration_constant: float = math.sqrt(2)) -> "MCTSNode":
        """Return the child with the highest UCB1 score.

        Args:
            exploration_constant: Passed directly to ucb1(). Higher values
                                  favor less-visited children.
        """
        return max(
            self.children,
            key=lambda child: ucb1(
                visits=child.visits,
                total_value=child.total_value,
                parent_visits=self.visits,
                c=exploration_constant,
            ),
        )

    def path_from_root(self) -> list[Move]:
        """Return the sequence of moves from root to this node.

        Used to reconstruct the game state at this node by replaying from
        a cloned root game, avoiding the need to store a game copy per node.
        """
        moves = []
        current: MCTSNode | None = self
        while current is not None and current.move is not None:
            moves.append(current.move)
            current = current.parent
        moves.reverse()
        return moves


class MCTSAgent(Agent):
    """An agent that uses Monte Carlo Tree Search to choose moves.

    Each call to choose_move builds a fresh search tree from the current
    game state, runs `simulations` iterations of select → expand → simulate
    → backpropagate, then returns the move with the most visits.

    Args:
        simulations: Number of simulations (iterations) to run per move.
                     Higher = stronger but slower. 200 is a reasonable default.
    """

    def __init__(self, simulations: int = 200):
        """
        Args:
            simulations: How many select/expand/simulate/backpropagate cycles
                         to run before picking the best move. More simulations
                         produce stronger play at the cost of more time.
        """
        self.simulations = simulations

    def choose_move(self, game: Game) -> Move:
        """Run MCTS from the current game state and return the best move found.

        The game passed in is never modified — we clone it once at the start
        and replay moves onto working copies per simulation.

        Raises:
            RuntimeError: If the game has no legal moves (should never happen
                          in a valid mid-round Azul state).
        """
        root_game = game.clone()
        legal_moves = root_game.legal_moves()
        if not legal_moves:
            raise RuntimeError("choose_move called with no legal moves available.")

        root = MCTSNode(move=None, parent=None, untried_moves=legal_moves)

        for _ in range(self.simulations):
            if root.is_fully_expanded() and not root.children:
                break  # terminal root — nothing to explore
            node, sim_game = self._select(root, root_game)
            node, sim_game = self._expand(node, sim_game)
            result = self._simulate(sim_game)
            self._backpropagate(node, result)

        if not root.children:
            # No simulations produced any children (e.g. simulations=0).
            # Fall back to the first legal move rather than crashing.
            return legal_moves[0]

        best = max(root.children, key=lambda node: node.visits)
        assert best.move is not None
        return best.move

    # ── The four MCTS phases ──────────────────────────────────────────────

    def _select(self, node: MCTSNode, root_game: Game) -> tuple[MCTSNode, Game]:
        """Walk down the tree to a node with untried moves (or a terminal).

        Descends greedily by UCB1 score until it reaches a node that still
        has unexplored moves or is a leaf. Returns the selected node and a
        game state advanced to that node's position.

        The game is reconstructed by replaying the path from root — one clone
        plus O(depth) make_move calls, with no further copies needed.
        """
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        sim_game = self._replay_to_node(node, root_game)
        return node, sim_game

    def _expand(self, node: MCTSNode, sim_game: Game) -> tuple[MCTSNode, Game]:
        """Add one new child by trying an unexplored move.

        Pops one move from untried_moves, applies it to sim_game, and creates
        a child node for that resulting state. Returns the new child and the
        now-advanced game (no additional clone needed — sim_game is mutated).

        If there are no untried moves (terminal or already fully expanded),
        returns the node unchanged.
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
        """Play out the current round randomly and return a result in [-1.0, 1.0].

        Stops at the end of the round rather than playing to game over. This
        cuts rollout length from ~150 moves to ~10–20, making each simulation
        much cheaper without meaningfully weakening move selection.

        The result is from the perspective of the player whose turn it is at
        the start of the simulation (sim_game.current_player_index):
            +1.0 = that player wins this round
            -1.0 = that player loses
             0.0 = tie

        Backpropagation handles flipping the perspective at each level, so
        this method never needs to know who the root player is.

        Args:
            sim_game: The game state to simulate from (mutated in place).
        """
        current_player_index = sim_game.current_player_index

        while True:
            legal = sim_game.legal_moves()
            if not legal:
                break
            move = random.choice(legal)
            sim_game.make_move(move)
            round_ended = sim_game.advance()
            if round_ended:
                break

        scores = [player.earned for player in sim_game.players]
        best_score = max(scores)
        winners = [index for index, score in enumerate(scores) if score == best_score]
        if len(winners) > 1:
            return 0.0
        return 1.0 if winners[0] == current_player_index else -1.0

    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Walk back up to the root, recording the simulation result at each node.

        The result is negated at each level because the parent is always the
        opponent: a result of +1.0 for the child (child's player wins) is -1.0
        for the parent (parent's player loses). This keeps every node's
        total_value in the perspective of the player whose turn it is there.

        Example with result=+1.0 (current player at leaf wins):
            leaf node         → stores +1.0
            parent node       → stores -1.0  (opponent's perspective: bad for them)
            grandparent node  → stores +1.0  (back to original player's perspective)
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += result
            result = -result  # negate to alternate perspective at each level
            current = current.parent

    # ── Internal helpers ──────────────────────────────────────────────────

    def _replay_to_node(self, node: MCTSNode, root_game: Game) -> Game:
        """Clone root_game and replay moves to reconstruct the game state at node."""
        sim_game = root_game.clone()
        for move in node.path_from_root():
            sim_game.make_move(move)
            sim_game.advance()
        return sim_game
