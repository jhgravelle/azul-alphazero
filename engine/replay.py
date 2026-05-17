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

from engine.game import Game, Move
from engine.game_recorder import GameRecord, _extract_seed_from_starting_state


def replay_to_move(record: GameRecord, move_index: int) -> Game:
    """Reconstruct a live Game from a GameRecord at a flat move index.

    Uses the seed-based approach: extract the seed from the first round's
    starting_state, create a single Game instance, then replay moves sequentially
    across all rounds. Rounds are set up via setup_round() on the first round,
    and subsequent rounds transition via advance() at round boundaries.
    """
    total_moves = sum(len(r.turns) for r in record.rounds)

    if move_index < 0 or move_index > total_moves:
        raise ValueError(f"move_index {move_index} out of range [0, {total_moves}]")

    # Extract seed from the first round's starting state
    if not record.rounds:
        return Game()

    seed = _extract_seed_from_starting_state(record.rounds[0].starting_state)

    # Create a single Game instance with the seed
    game = Game(seed=seed)
    game.setup_round()

    # Return immediately if targeting the initial state
    if move_index == 0:
        return game

    moves_replayed = 0

    # Replay all moves across all rounds
    for round_record in record.rounds:
        for turn_record in round_record.turns:
            move = Move.from_str(turn_record.move)
            game.make_move(move)
            game.advance()
            moves_replayed += 1

            if moves_replayed == move_index:
                return game

    return game
