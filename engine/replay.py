# engine/replay.py


"""Utilities for replaying recorded Azul games.

The single public function, replay_to_move, reconstructs a live Game object
from a GameRecord at any point in the game's history. This is the foundation
for the MCTS inspector: given a recorded game state, build a fresh SearchTree
rooted there and run simulations.

move_index semantics
--------------------
move_index is a flat count of moves across all rounds, starting at zero.

    move_index=0  →  game after setup_round() for round 1, before any moves
    move_index=1  →  game after the first move of round 1
    move_index=N  →  game after move N-1 has been played

After the last move of a round, setup_round() is called automatically so the
returned Game is always in a playable state (factories filled, ready for the
next move or — at move_index=total — ready for score_game()).

Raises
------
ValueError  if move_index < 0 or move_index > total moves in the record.
"""

from engine.constants import Tile
from engine.game import Game, Move
from engine.game_recorder import GameRecord


def replay_to_move(record: GameRecord, move_index: int) -> Game:
    """Reconstruct a live Game from a GameRecord at a flat move index.

    Args:
        record:     A fully loaded GameRecord.
        move_index: How many moves to replay (0 = initial state).

    Returns:
        A Game object whose state reflects exactly move_index moves played.
        If move_index falls at a round boundary, setup_round() has already
        been called so the returned game has factories filled and is ready
        to play.

    Raises:
        ValueError: if move_index is negative or exceeds total moves.
    """
    total_moves = sum(len(r.moves) for r in record.rounds)

    if move_index < 0 or move_index > total_moves:
        raise ValueError(f"move_index {move_index} out of range [0, {total_moves}]")

    game = Game()
    moves_replayed = 0

    for round_record in record.rounds:
        explicit_factories = [
            [Tile[name] for name in factory] for factory in round_record.factories
        ]
        game.setup_round(factories=explicit_factories)

        if moves_replayed == move_index:
            # Reached the target before playing any move this round —
            # return with factories already filled.
            return game

        for move_record in round_record.moves:
            move = Move(
                source=move_record.source,
                tile=Tile[move_record.tile],
                destination=move_record.destination,
            )
            game.make_move(move)
            moves_replayed += 1

            if moves_replayed == move_index:
                # If this move ended the round, set up the next round so the
                # returned game is always in a playable state.
                game.advance(skip_setup=True)
                _setup_next_round(record, game, moves_replayed, total_moves)
                return game

            game.advance()

    # move_index == total_moves: all moves replayed, game not yet scored.
    return game


def _setup_next_round(
    record: GameRecord,
    game: Game,
    moves_replayed: int,
    total_moves: int,
) -> None:
    """Set up the next round using the recorded factory layout, if available.

    When move_index lands exactly on a round boundary, the caller has the game
    in a sources-empty state. We set up the next round using the recorded
    factories so the returned game matches what actually happened, rather than
    drawing randomly from the bag.

    Does nothing if we are at the very end of the game.
    """
    if moves_replayed >= total_moves:
        return

    # Find which round comes next by counting moves.
    cursor = 0
    for round_record in record.rounds:
        cursor += len(round_record.moves)
        if cursor == moves_replayed:
            # round_record is the round that just ended.
            # The next round_record (if any) has the factories we want.
            next_round_idx = record.rounds.index(round_record) + 1
            if next_round_idx < len(record.rounds):
                next_round = record.rounds[next_round_idx]
                explicit_factories = [
                    [Tile[name] for name in factory] for factory in next_round.factories
                ]
                game.setup_round(factories=explicit_factories)
            return
