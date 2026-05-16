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
        depth: int = 2,
        threshold: int = 6,
    ) -> None:
        self.depth = depth
        self.threshold = threshold
        self._nodes: int = 0

    def _effective_depth(self, game: Game) -> int:
        """Return search depth based on remaining source-colors this round.

        When the board is wide (many source-colors remaining), cap depth to
        avoid combinatorial explosion. When narrow (few source-colors), use
        the source count itself as depth — searching as deep as there are
        moves remaining this round.
        """
        sources = game.total_source_count
        if sources > self.threshold:
            return self.depth
        return sources

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
