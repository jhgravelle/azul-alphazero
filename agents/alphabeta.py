# agents/alphabeta.py
import math
from engine.game import Game, Move
from engine.scoring import earned_score_unclamped
from agents.base import Agent

_POLICY_TEMPERATURE = 1.0


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
        depths: tuple[int, int, int] = (2, 3, 5),
        thresholds: tuple[int, int] = (10, 7),
    ) -> None:
        self.depths = depths
        self.thresholds = thresholds
        self._nodes: int = 0
        self._root_move_scores: list[tuple[Move, float]] = []

    def _effective_depth(self, game: Game) -> int:
        legal_count = len(game.legal_moves())
        if legal_count > self.thresholds[0]:
            return self.depths[0]
        elif legal_count > self.thresholds[1]:
            return self.depths[1]
        else:
            return self.depths[2]

    def choose_move(self, game: Game) -> Move:
        self._nodes = 0
        self._root_move_scores = []
        legal = game.legal_moves()
        depth = self._effective_depth(game)
        root_player = game.state.current_player

        self._root_move_scores = self._score_all_root_moves(
            game, legal, depth, root_player
        )

        return max(self._root_move_scores, key=lambda pair: pair[1])[0]

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
        root_player: int,
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
                root_player,
                float("-inf"),
                float("inf"),
            )
            scored.append((move, score))
        return scored

    def _immediate_score(self, game: Game, move: Move, root_player: int) -> float:
        moving_player = game.state.current_player
        before = earned_score_unclamped(game.state.players[moving_player])
        child = game.clone()
        child.make_move(move)
        after = earned_score_unclamped(child.state.players[moving_player])
        delta = after - before
        return delta if moving_player == root_player else -delta

    def _move_order_key(self, move: Move, game: Game, root_player: int) -> float:
        moving_player = game.state.current_player
        if move.destination == -2:
            return -10.0 if moving_player == root_player else 10.0
        capacity = move.destination + 1
        line = game.state.players[moving_player].pattern_lines[move.destination]
        fills_line = len(line) + 1 >= capacity
        if fills_line:
            return 5.0 if moving_player == root_player else -5.0
        return 0.0

    def _alphabeta(
        self,
        game: Game,
        move: Move,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
    ) -> float:
        self._nodes += 1
        moving_player = game.state.current_player
        before = earned_score_unclamped(game.state.players[moving_player])

        child = game.clone()
        child.make_move(move)
        after = earned_score_unclamped(child.state.players[moving_player])
        delta = after - before
        immediate = delta if moving_player == root_player else -delta

        round_ended = child.advance(skip_setup=True)

        if child.is_game_over() or round_ended or depth <= 1:
            return immediate

        legal = child.legal_moves()
        if not legal:
            return immediate

        next_player = child.state.current_player
        maximizing = next_player == root_player

        ordered = sorted(
            legal,
            key=lambda m: self._move_order_key(m, child, root_player),
            reverse=maximizing,
        )

        if maximizing:
            best = float("-inf")
            for m in ordered:
                score = immediate + self._alphabeta(
                    child,
                    m,
                    depth - 1,
                    root_player,
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
                    root_player,
                    alpha - immediate,
                    beta - immediate,
                )
                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break
            return best
