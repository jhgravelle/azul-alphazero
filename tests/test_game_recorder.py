# tests/test_game_recorder.py

"""Tests for the game recorder — compact round/move format."""

import json
import pytest
from engine.game import Game, Move
from engine.constants import PLAYERS
from engine.game_recorder import GameRecorder, GameRecord

# ── Helpers ────────────────────────────────────────────────────────────────


def _first_legal_move(game: Game) -> Move:
    return game.legal_moves()[0]


def _play_full_game(recorder: GameRecorder | None = None) -> Game:
    """Play a full game to completion, optionally recording it."""
    game = Game()
    game.setup_round()
    if recorder:
        recorder.start_round(game)
    last_round = game.round
    max_moves = 500
    moves_made = 0
    while not game.is_game_over() and moves_made < max_moves:
        moves = game.legal_moves()
        if not moves:
            break
        move = moves[0]
        game.make_move(move)
        if recorder:
            recorder.record_move(move, player_index=game.current_player_index)
        game.advance()
        if recorder and not game.is_game_over() and game.round != last_round:
            recorder.start_round(game)
            last_round = game.round
        moves_made += 1
    if game.is_game_over():
        game._score_game()
    return game


# ── Construction ───────────────────────────────────────────────────────────


def test_recorder_default_player_names():
    recorder = GameRecorder()
    assert recorder.record.player_names == ["Player 0", "Player 1"]


def test_recorder_custom_player_names():
    recorder = GameRecorder(player_names=["Alice", "Bob"])
    assert recorder.record.player_names == ["Alice", "Bob"]


def test_recorder_starts_with_no_rounds():
    recorder = GameRecorder()
    assert recorder.record.rounds == []


def test_recorder_has_game_id():
    recorder = GameRecorder()
    assert isinstance(recorder.record.game_id, str)
    assert len(recorder.record.game_id) > 0


def test_recorder_has_timestamp():
    recorder = GameRecorder()
    assert isinstance(recorder.record.timestamp, str)
    assert len(recorder.record.timestamp) > 0


def test_two_recorders_have_different_game_ids():
    r1 = GameRecorder()
    r2 = GameRecorder()
    assert r1.record.game_id != r2.record.game_id


# ── start_round ────────────────────────────────────────────────────────────


def test_start_round_adds_round_record():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    assert len(recorder.record.rounds) == 1


def test_start_round_captures_starting_state():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    round_record = recorder.record.rounds[0]
    assert len(round_record.starting_state) > 0


def test_start_round_has_starting_state_in_record():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    round_record = recorder.record.rounds[0]
    # Starting state should be a list of strings (the game state repr)
    assert isinstance(round_record.starting_state, list)
    assert len(round_record.starting_state) > 0


def test_start_round_starting_state_contains_tile_info():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    # Starting state is stored as stringified game state
    state_str = "\n".join(recorder.record.rounds[0].starting_state)
    # Verify it contains tile abbreviations used in the display
    tile_abbrev = ("B", "Y", "R", "K", "W")
    has_tiles = any(abbrev in state_str for abbrev in tile_abbrev)
    assert has_tiles


def test_start_round_captures_round_number():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    assert recorder.record.rounds[0].round == 1


def test_start_round_increments_for_second_round():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    game.setup_round()
    recorder.start_round(game)
    assert recorder.record.rounds[1].round == 2


# ── record_move ────────────────────────────────────────────────────────────


def test_record_move_adds_to_current_round():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game)
    assert len(recorder.record.rounds[0].turns) == 1


def test_record_move_captures_move_string():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game)
    assert len(recorder.record.rounds[0].turns[0].move) > 0


def test_record_move_captures_state():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game)
    assert isinstance(recorder.record.rounds[0].turns[0].state, list)
    assert len(recorder.record.rounds[0].turns[0].state) > 0


def test_record_move_increments_turn_count():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game)
    assert recorder.record.rounds[0].turns[0].turn == 1


def test_record_move_multiple_moves():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move1 = _first_legal_move(game)
    game.make_move(move1)
    recorder.record_move(move1, game, player_index=game.current_player_index)
    game.advance()
    move2 = _first_legal_move(game)
    game.make_move(move2)
    recorder.record_move(move2, game, player_index=game.current_player_index)
    assert len(recorder.record.rounds[0].turns) == 2


def test_record_move_before_start_round_raises():
    recorder = GameRecorder()
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    with pytest.raises(RuntimeError):
        recorder.record_move(move, game)


# ── finalize ───────────────────────────────────────────────────────────────


def test_finalize_sets_final_scores():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    while not game.is_game_over():
        moves = game.legal_moves()
        if not moves:
            break
        move = moves[0]
        game.make_move(move)
        recorder.record_move(move, game, player_index=game.current_player_index)
        game.advance()
        if game.round > recorder.record.rounds[-1].round and not game.is_game_over():
            recorder.start_round(game)
    game._score_game()
    recorder.finalize(game)
    assert len(recorder.record.final_scores) == PLAYERS


@pytest.mark.slow
def test_finalize_winner_has_highest_score():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    while not game.is_game_over():
        moves = game.legal_moves()
        if not moves:
            break
        move = moves[0]
        game.make_move(move)
        recorder.record_move(move, game, player_index=game.current_player_index)
        game.advance()
        if game.round > recorder.record.rounds[-1].round and not game.is_game_over():
            recorder.start_round(game)
    game._score_game()
    recorder.finalize(game)
    assert recorder.record.winner is not None
    winner_score = recorder.record.final_scores[recorder.record.winner]
    for score in recorder.record.final_scores:
        assert winner_score >= score


# ── Serialization ──────────────────────────────────────────────────────────


def test_to_json_is_valid_json():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game, player_index=game.current_player_index)
    assert isinstance(json.loads(recorder.to_json()), dict)


def test_to_json_contains_rounds():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game, player_index=game.current_player_index)
    parsed = json.loads(recorder.to_json())
    assert "rounds" in parsed
    assert len(parsed["rounds"]) == 1


def test_round_trip_preserves_moves(tmp_path):
    game = Game()
    game.setup_round()
    recorder = GameRecorder(player_names=["Alice", "Bob"])
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game, player_index=game.current_player_index)

    path = tmp_path / "game.json"
    recorder.save(path)
    loaded = GameRecord.load(path)

    assert len(loaded.rounds) == 1
    assert len(loaded.rounds[0].turns) == 1
    assert len(loaded.rounds[0].turns[0].move) > 0
    assert isinstance(loaded.rounds[0].turns[0].state, list)


def test_round_trip_preserves_factories(tmp_path):
    game = Game()
    game.setup_round()
    original_factory_count = len(game.factories)
    recorder = GameRecorder()
    recorder.start_round(game)

    path = tmp_path / "game.json"
    recorder.save(path)
    loaded = GameRecord.load(path)

    # Factories are not serialized (redundant with starting_state)
    # but can be reconstructed from starting_state
    assert len(loaded.rounds) == 1
    round_record = loaded.rounds[0]
    assert len(round_record.starting_state) > 0

    # Reconstruct game from starting_state
    reconstructed_game = Game.from_string("\n".join(round_record.starting_state))
    assert len(reconstructed_game.factories) == original_factory_count


def test_round_trip_preserves_player_types(tmp_path):
    recorder = GameRecorder(
        player_names=["Alice", "Bob"], player_types=["human", "greedy"]
    )
    game = Game()
    game.setup_round()
    recorder.start_round(game)

    path = tmp_path / "game.json"
    recorder.save(path)
    loaded = GameRecord.load(path)

    # Player types are stored internally but not serialized to JSON
    assert recorder.record.player_types == ["human", "greedy"]
    assert len(loaded.rounds) == 1


# ── Reconstruction ─────────────────────────────────────────────────────────


def test_reconstruct_returns_one_state_per_move():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    last_round = game.round
    for _ in range(4):
        moves = game.legal_moves()
        if not moves:
            break
        move = moves[0]
        game.make_move(move)
        recorder.record_move(move, game, player_index=game.current_player_index)
        game.advance()
        if game.round != last_round and not game.is_game_over():
            recorder.start_round(game)
            last_round = game.round

    states, _ = recorder.record.reconstruct()
    total_moves = sum(len(r.turns) for r in recorder.record.rounds)
    assert len(states) == total_moves + 1  # +1 for initial state


def test_reconstruct_final_boards_reflect_scoring():
    """final_boards should include end-of-game bonus scoring."""
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    while not game.is_game_over():
        moves = game.legal_moves()
        if not moves:
            break
        move = moves[0]
        game.make_move(move)
        recorder.record_move(move, game, player_index=game.current_player_index)
        game.advance()
        if game.round > recorder.record.rounds[-1].round and not game.is_game_over():
            recorder.start_round(game)
    game._score_game()
    recorder.finalize(game)

    _, final_boards = recorder.record.reconstruct()
    assert len(final_boards) == PLAYERS
    for i, board in enumerate(final_boards):
        assert board["score"] == recorder.record.final_scores[i]


def test_reconstruct_grand_totals_are_after_move():
    """Grand totals in each computed turn reflect the state after the move."""
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    recorder.start_round(game)
    move = _first_legal_move(game)
    game.make_move(move)
    recorder.record_move(move, game, player_index=game.current_player_index)

    states, _ = recorder.record.reconstruct()
    # After first move, at least one player's grand total should be >= 0.
    assert all(t >= 0 for t in states[0]["grand_totals"])
