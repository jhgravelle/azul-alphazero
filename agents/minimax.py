# agents/minimax.py

from engine.game import Game, Move
from agents.base import Agent


class MinimaxAgent(Agent):
    """Depth-limited minimax agent using immediate score as evaluation.

    Stops at round boundaries and game over — does not cross into the
    next round's random factory setup.
    """

    def __init__(
        self,
        depths: tuple[int, int, int] = (2, 3, 4),
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
        root_player_index = game.current_player_index
        depth = self._effective_depth(game)
        best_move = legal[0]
        best_score = float("-inf")
        for move in legal:
            score = self._minimax(game, move, depth, root_player_index)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(
        self, game: Game, move: Move, depth: int, root_player_index: int
    ) -> float:
        self._nodes += 1
        moving_player_index = game.current_player_index
        before = game.players[moving_player_index].earned

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
        if next_player_index == root_player_index:
            best = max(
                self._minimax(child, m, depth - 1, root_player_index) for m in legal
            )
        else:
            best = min(
                self._minimax(child, m, depth - 1, root_player_index) for m in legal
            )
        return immediate + best
