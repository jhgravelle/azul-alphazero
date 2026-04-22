# agents/alphabeta.py
from engine.game import Game, Move
from engine.scoring import earned_score_unclamped
from agents.base import Agent


class AlphaBetaAgent(Agent):
    """Depth-limited minimax with alpha-beta pruning and move ordering.

    Identical decisions to MinimaxAgent at the same depth, but faster
    due to pruning — allowing deeper search in the same time budget.
    """

    def __init__(
        self,
        depths: tuple[int, int, int] = (2, 3, 5),
        thresholds: tuple[int, int] = (10, 7),
    ) -> None:
        self.depths = depths
        self.thresholds = thresholds
        self._nodes: int = 0

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
        legal = game.legal_moves()
        root_player = game.state.current_player
        depth = self._effective_depth(game)
        best_move = legal[0]
        alpha = float("-inf")
        beta = float("inf")
        for move in legal:
            score = self._alphabeta(game, move, depth, root_player, alpha, beta)
            if score > alpha:
                alpha = score
                best_move = move
        return best_move

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

        # ordered = legal
        # ordered = sorted(
        #     legal,
        #     key=lambda m: self._immediate_score(child, m, root_player),
        #     reverse=maximizing,
        # )
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
