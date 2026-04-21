# agents/minimax.py

from engine.game import Game, Move
from engine.scoring import earned_score_unclamped
from agents.base import Agent


class MinimaxAgent(Agent):
    """Depth-limited minimax agent using immediate score as evaluation.

    Stops at round boundaries and game over — does not cross into the
    next round's random factory setup.
    """

    def __init__(self, depth: int = 3) -> None:
        self.depth = depth

    def choose_move(self, game: Game) -> Move:
        legal = game.legal_moves()
        root_player = game.state.current_player
        best_move = legal[0]
        best_score = float("-inf")
        for move in legal:
            score = self._minimax(game, move, self.depth, root_player)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, game: Game, move: Move, depth: int, root_player: int) -> float:
        moving_player = game.state.current_player
        before = earned_score_unclamped(game.state.players[moving_player])

        child = game.clone()
        child.make_move(move)
        child.advance()

        after = earned_score_unclamped(child.state.players[moving_player])
        delta = after - before
        immediate = delta if moving_player == root_player else -delta

        if child.is_game_over() or child.is_round_over() or depth <= 1:
            return immediate

        legal = child.legal_moves()
        if not legal:
            return immediate

        next_player = child.state.current_player
        if next_player == root_player:
            best = max(self._minimax(child, m, depth - 1, root_player) for m in legal)
        else:
            best = min(self._minimax(child, m, depth - 1, root_player) for m in legal)
        return immediate + best
