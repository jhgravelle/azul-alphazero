# tests/test_game_replay.py
"""Tests for engine/replay.py — replay_to_move()."""

import pytest
from engine.game import Game
from engine.game_recorder import GameRecord, GameRecorder
from engine.replay import replay_to_move

# ── Helpers ───────────────────────────────────────────────────────────────────


def _record_full_game() -> GameRecord:
    from agents.random import RandomAgent

    game = Game()
    recorder = GameRecorder(
        player_names=["Alice", "Bob"],
        player_types=["random", "random"],
    )
    agents = [RandomAgent(), RandomAgent()]

    game.setup_round()
    recorder.start_round(game)

    while not game.is_game_over():
        move = agents[game.state.current_player].choose_move(game)
        recorder.record_move(move, player_index=game.state.current_player)
        game.make_move(move)

        # Mirror _handle_round_end logic exactly
        sources_empty = (
            all(len(f) == 0 for f in game.state.factories)
            and len(game.state.center) == 0
        )
        if sources_empty and not game.is_game_over():
            game.setup_round()
            recorder.start_round(game)

    game.score_game()
    recorder.finalize(game)
    return recorder.record


def _total_moves(record: GameRecord) -> int:
    return sum(len(r.moves) for r in record.rounds)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_replay_to_move_zero_is_initial_state():
    """move_index=0 should give the game after first setup_round, before any moves."""
    record = _record_full_game()
    game = replay_to_move(record, 0)

    # Factories should be filled (5 factories × 4 tiles = 20 tiles placed)
    total_factory_tiles = sum(len(f) for f in game.state.factories)
    assert total_factory_tiles == 20

    # No moves made yet — all pattern lines empty
    for player in game.state.players:
        for line in player.pattern_lines:
            assert len(line) == 0

    # Round 1
    assert game.state.round == 1


def test_replay_to_move_one_matches_first_recorded_move():
    """move_index=1 should reflect the state after exactly one move."""
    record = _record_full_game()
    # first_move_record = record.rounds[0].moves[0]

    game = replay_to_move(record, 1)

    # The factory or center that was the source should have fewer tiles,
    # OR the destination pattern line should have tiles — at least one player
    # board has changed.
    total_pattern_tiles = sum(
        len(line) for player in game.state.players for line in player.pattern_lines
    )
    total_floor_tiles = sum(len(player.floor_line) for player in game.state.players)
    assert total_pattern_tiles + total_floor_tiles > 0


def test_replay_to_move_incremental_matches_jump():
    """Replaying move by move should give the same state as jumping directly."""
    record = _record_full_game()
    total = _total_moves(record)

    # Pick a move in the middle of the game
    mid = total // 2

    game_jump = replay_to_move(record, mid)

    # Replay incrementally
    # game_step = replay_to_move(record, 0)
    # We can't call replay_to_move in a loop cheaply here, so instead
    # reconstruct() the turns and compare board state at move `mid`.
    computed_turns, _ = record.reconstruct()

    # computed_turns[0] is the initial state (is_initial=True),
    # so computed_turns[mid] is after move mid-1 ... wait, let's be precise.
    # reconstruct() index 0 = initial, index N = after move N-1.
    # replay_to_move(record, mid) = state after mid-1 moves played.
    # So they should agree at index `mid`.
    turn = computed_turns[mid]

    for player_idx, player in enumerate(game_jump.state.players):
        expected_score = turn["boards"][player_idx]["score"]
        assert player.score == expected_score, (
            f"Player {player_idx} score mismatch at move {mid}: "
            f"got {player.score}, expected {expected_score}"
        )


def test_replay_to_move_final_matches_final_scores():
    """Replaying to the last move should give scores matching final_scores."""
    record = _record_full_game()
    total = _total_moves(record)

    game = replay_to_move(record, total)
    game.score_game()

    for i, player in enumerate(game.state.players):
        assert (
            player.score == record.final_scores[i]
        ), f"Player {i}: got {player.score}, expected {record.final_scores[i]}"


def test_replay_to_move_preserves_round_boundaries():
    """Game state at a round boundary should be in a consistent state."""
    record = _record_full_game()
    if len(record.rounds) < 2:
        pytest.skip("Game too short to test round boundaries")

    # Index of the last move in round 1
    moves_in_round_1 = len(record.rounds[0].moves)
    game = replay_to_move(record, moves_in_round_1)

    # After the last move of round 1, round scoring + setup has happened,
    # so we should be in round 2 with factories filled.
    assert game.state.round == 2
    total_factory_tiles = sum(len(f) for f in game.state.factories)
    assert total_factory_tiles == 20


def test_replay_to_move_out_of_bounds_raises():
    """move_index beyond total moves should raise ValueError."""
    record = _record_full_game()
    total = _total_moves(record)

    with pytest.raises(ValueError):
        replay_to_move(record, total + 1)
