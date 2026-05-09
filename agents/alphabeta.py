# agents/alphabeta.py
import math
import random
from engine.game import Game, Move
from agents.base import Agent

_POLICY_TEMPERATURE = 1.0
_EXPLORATION_TEMPERATURE = 0.3
_END_OF_ROUND_TEMPERATURE = 1.0
_END_OF_ROUND_SOURCES_THRESHOLD = 2


def _softmax_distribution(
    scored_moves: list[tuple[Move, float]],
    temperature: float,
) -> list[tuple[Move, float]]:
    """Convert a list of (move, score) pairs into a probability distribution.

    Applies softmax with the given temperature. Shifts by the max score
    before exponentiation for numerical stability — the best move always
    gets exp(0) = 1.0 and all others get a fraction.

    Equal scores produce equal probabilities. Higher temperature produces
    a softer (more uniform) distribution; lower temperature peaks more
    sharply on the best move.
    """
    scores = [score for _, score in scored_moves]
    max_score = max(scores)
    exps = [math.exp((score - max_score) / temperature) for score in scores]
    total = sum(exps)
    return [(move, exp_val / total) for (move, _), exp_val in zip(scored_moves, exps)]


def _sample_from_distribution(
    distribution: list[tuple[Move, float]],
    rng: random.Random,
) -> Move:
    moves = [move for move, _ in distribution]
    weights = [prob for _, prob in distribution]
    return rng.choices(moves, weights=weights, k=1)[0]


class AlphaBetaAgent(Agent):
    """Depth-limited minimax with alpha-beta pruning and move ordering.

    Identical decisions to MinimaxAgent at the same depth, but faster
    due to pruning — allowing deeper search in the same time budget.

    policy_distribution() returns a softmax over root move scores from
    the most recent choose_move() call. This gives the training pipeline
    a soft, search-informed distribution rather than uniform over all
    legal moves.
    """

    def __init__(
        self,
        depth: int = 2,
        threshold: int = 6,
        exploration_temperature: float = _EXPLORATION_TEMPERATURE,
    ) -> None:
        self.depth = depth
        self.threshold = threshold
        self.exploration_temperature = exploration_temperature
        self._rng = random.Random()  # independent RNG per agent instance
        self._nodes: int = 0
        self._root_move_scores: list[tuple[Move, float]] = []

    def _effective_depth(self, game: Game) -> int:
        """Return search depth based on remaining source-colors this round.

        When the board is wide (many source-colors remaining), cap depth to
        avoid combinatorial explosion. When narrow (few source-colors), use
        the source count itself as depth — searching as deep as there are
        moves remaining this round.
        """
        sources = sum(
            source_count for _, source_count in game.tile_availability().values()
        )
        if sources > self.threshold:
            return self.depth
        return sources

    def _pick_move(self, game: Game) -> Move:
        """Sample a move from the scored distribution using the appropriate temperature.

        Uses end-of-round temperature (fully exploratory) when only a few
        sources remain, and exploration temperature (mostly smart) otherwise.
        """
        if self.exploration_temperature == 0.0:
            return max(self._root_move_scores, key=lambda pair: pair[1])[0]

        sources = sum(
            source_count for _, source_count in game.tile_availability().values()
        )
        temperature = (
            _END_OF_ROUND_TEMPERATURE
            if sources <= _END_OF_ROUND_SOURCES_THRESHOLD
            else self.exploration_temperature
        )
        distribution = _softmax_distribution(self._root_move_scores, temperature)
        return _sample_from_distribution(distribution, self._rng)

    def choose_move(self, game: Game) -> Move:
        self._nodes = 0
        self._root_move_scores = []
        legal = game.legal_moves()
        depth = self._effective_depth(game)
        root_player_index = game.current_player_index

        self._root_move_scores = self._score_all_root_moves(
            game, legal, depth, root_player_index
        )

        return self._pick_move(game)

    def policy_distribution(self, game: Game) -> list[tuple[Move, float]]:
        """Softmax over root move scores from the most recent choose_move call.

        Falls back to uniform over legal moves if choose_move has not been
        called yet for this position — the base class handles that case.
        """
        if not self._root_move_scores:
            return super().policy_distribution(game)
        return _softmax_distribution(
            self._root_move_scores, temperature=_POLICY_TEMPERATURE
        )

    def _score_all_root_moves(
        self,
        game: Game,
        legal: list[Move],
        depth: int,
        root_player_index: int,
    ) -> list[tuple[Move, float]]:
        """Score every root move independently with a full alpha-beta window.

        No pruning between root moves — each is evaluated with
        alpha=-inf, beta=+inf so every move receives a fair score
        for use in policy_distribution. Child nodes still prune normally.

        Returns moves in the same order as the input legal list.
        """
        scored = []
        for move in legal:
            score = self._alphabeta(
                game,
                move,
                depth,
                root_player_index,
                float("-inf"),
                float("inf"),
            )
            scored.append((move, score))
        return scored

    def _move_order_key(self, move: Move, game: Game, root_player_index: int) -> float:
        moving_player_index = game.current_player_index
        if move.destination == -2:
            return -10.0 if moving_player_index == root_player_index else 10.0
        from engine.constants import CAPACITY, COLUMN_FOR_TILE_IN_ROW

        capacity = CAPACITY[move.destination]
        col = COLUMN_FOR_TILE_IN_ROW[move.tile][move.destination]
        filled = game.current_player.pattern_grid[move.destination][col]
        fills_line = filled + 1 >= capacity
        if fills_line:
            return 5.0 if moving_player_index == root_player_index else -5.0
        return 0.0

    def _alphabeta(
        self,
        game: Game,
        move: Move,
        depth: int,
        root_player_index: int,
        alpha: float,
        beta: float,
    ) -> float:
        self._nodes += 1
        moving_player_index = game.current_player_index
        before = game.current_player.earned

        child = game.clone()
        child.make_move(move)
        after = child.players[moving_player_index].earned
        delta = after - before
        immediate = delta if moving_player_index == root_player_index else -delta

        round_ended = child.advance(skip_setup=True)

        if child.is_game_over() or round_ended or depth <= 1:
            return immediate

        legal = child.legal_moves()
        if not legal:
            return immediate

        next_player_index = child.current_player_index
        maximizing = next_player_index == root_player_index

        ordered = sorted(
            legal,
            key=lambda m: self._move_order_key(m, child, root_player_index),
            reverse=maximizing,
        )

        if maximizing:
            best = float("-inf")
            for m in ordered:
                score = immediate + self._alphabeta(
                    child,
                    m,
                    depth - 1,
                    root_player_index,
                    alpha - immediate,
                    beta - immediate,
                )
                best = max(best, score)
                alpha = max(alpha, best)
                if beta <= alpha:
                    break
            return best
        else:
            best = float("inf")
            for m in ordered:
                score = immediate + self._alphabeta(
                    child,
                    m,
                    depth - 1,
                    root_player_index,
                    alpha - immediate,
                    beta - immediate,
                )
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best
