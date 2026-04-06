# tests/test_game_recorder.py

"""Tests for the game recorder."""

import json

# import pytest
from engine.game import Game, Move  # , CENTER
from engine.constants import PLAYERS, Tile
from engine.game_recorder import GameRecorder, GameRecord  # , TurnRecord


# ── Construction ───────────────────────────────────────────────────────────


def test_recorder_default_player_names():
    recorder = GameRecorder()
    assert recorder.player_names == ["Player 0", "Player 1"]


def test_recorder_custom_player_names():
    recorder = GameRecorder(player_names=["Alice", "Bob"])
    assert recorder.player_names == ["Alice", "Bob"]


def test_recorder_starts_with_no_turns():
    recorder = GameRecorder()
    assert recorder.record.turns == []


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


# ── record_turn ────────────────────────────────────────────────────────────


def _first_legal_move(game: Game) -> Move:
    return game.legal_moves()[0]


def test_record_turn_adds_one_turn():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    assert len(recorder.record.turns) == 1


def test_record_turn_captures_current_player():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    turn = recorder.record.turns[0]
    assert turn.player_index == game.state.current_player


def test_record_turn_captures_move_source():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    assert recorder.record.turns[0].move_source == move.source


def test_record_turn_captures_move_tile():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    assert recorder.record.turns[0].move_tile == move.tile.name


def test_record_turn_captures_move_destination():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    assert recorder.record.turns[0].move_destination == move.destination


def test_record_turn_captures_board_state_for_both_players():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    turn = recorder.record.turns[0]
    assert len(turn.board_states) == PLAYERS


def test_record_turn_board_state_has_score():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    assert "score" in board_state


def test_record_turn_board_state_has_pattern_lines():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    assert "pattern_lines" in board_state
    assert len(board_state["pattern_lines"]) == 5


def test_record_turn_board_state_has_wall():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    assert "wall" in board_state
    assert len(board_state["wall"]) == 5
    assert len(board_state["wall"][0]) == 5


def test_record_turn_board_state_has_floor_line():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    assert "floor_line" in board_state


def test_record_turn_board_state_uses_tile_names_not_enum_values():
    """Tile names like 'BLUE' should appear in the JSON, not enum integers."""
    game = Game()
    game.setup_round()
    player = game.state.players[0]
    player.wall[0][0] = Tile.BLUE
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    wall_flat = [cell for row in board_state["wall"] for cell in row]
    assert "BLUE" in wall_flat


def test_record_turn_empty_wall_cell_is_none():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    wall_flat = [cell for row in board_state["wall"] for cell in row]
    assert None in wall_flat


def test_record_turn_pattern_line_uses_tile_names():
    """A non-empty pattern line should store tile names, not enum values."""
    game = Game()
    game.setup_round()
    player = game.state.players[0]
    player.pattern_lines[1] = [Tile.RED]
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    assert board_state["pattern_lines"][1] == ["RED"]


def test_record_turn_empty_pattern_line_is_empty_list():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    board_state = recorder.record.turns[0].board_states[0]
    assert board_state["pattern_lines"][0] == []


def test_record_turn_with_no_analysis():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    assert recorder.record.turns[0].analysis is None


def test_record_turn_with_analysis():
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    analysis = {"value_estimate": 0.6, "simulations": 100}
    recorder.record_turn(game, move, analysis=analysis)
    assert recorder.record.turns[0].analysis == analysis


def test_record_turn_captures_state_before_move_is_applied():
    """Score should reflect the state BEFORE the move is applied."""
    game = Game()
    game.setup_round()
    recorder = GameRecorder()
    move = _first_legal_move(game)
    score_before = game.state.players[0].score
    recorder.record_turn(game, move)
    game.make_move(move)
    recorded_score = recorder.record.turns[0].board_states[0]["score"]
    assert recorded_score == score_before


# ── finalize ───────────────────────────────────────────────────────────────


def _play_full_game(recorder: GameRecorder | None = None) -> Game:
    """Play a full game to completion using legal moves."""
    game = Game()
    game.setup_round()
    max_moves = 500
    moves_made = 0
    while not game.is_game_over() and moves_made < max_moves:
        moves = game.legal_moves()
        if not moves:
            game.score_round()
            if not game.is_game_over():
                game.setup_round()
            continue
        move = moves[0]
        if recorder:
            recorder.record_turn(game, move)
        game.make_move(move)
        moves_made += 1
    if game.is_game_over():
        game.score_game()
    return game


def test_finalize_sets_final_scores():
    recorder = GameRecorder()
    game = _play_full_game(recorder)
    recorder.finalize(game)
    assert len(recorder.record.final_scores) == PLAYERS


def test_finalize_scores_match_game_state():
    recorder = GameRecorder()
    game = _play_full_game(recorder)
    recorder.finalize(game)
    for i, player in enumerate(game.state.players):
        assert recorder.record.final_scores[i] == player.score


def test_finalize_sets_winner():
    recorder = GameRecorder()
    game = _play_full_game(recorder)
    recorder.finalize(game)
    assert recorder.record.winner is not None


def test_finalize_winner_has_highest_score():
    recorder = GameRecorder()
    game = _play_full_game(recorder)
    recorder.finalize(game)
    assert recorder.record.winner is not None
    winner_score = recorder.record.final_scores[recorder.record.winner]
    for score in recorder.record.final_scores:
        assert winner_score >= score


# ── Serialization ──────────────────────────────────────────────────────────


def test_to_json_returns_string():
    recorder = GameRecorder()
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    assert isinstance(recorder.to_json(), str)


def test_to_json_is_valid_json():
    recorder = GameRecorder()
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    parsed = json.loads(recorder.to_json())
    assert isinstance(parsed, dict)


def test_to_json_contains_game_id():
    recorder = GameRecorder()
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    parsed = json.loads(recorder.to_json())
    assert "game_id" in parsed


def test_to_json_contains_turns():
    recorder = GameRecorder()
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    parsed = json.loads(recorder.to_json())
    assert "turns" in parsed
    assert len(parsed["turns"]) == 1


def test_to_json_contains_player_names():
    recorder = GameRecorder(player_names=["Alice", "Bob"])
    parsed = json.loads(recorder.to_json())
    assert parsed["player_names"] == ["Alice", "Bob"]


def test_round_trip_preserves_turns(tmp_path):
    recorder = GameRecorder(player_names=["Alice", "Bob"])
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)
    recorder.finalize(game)

    path = tmp_path / "game.json"
    recorder.save(path)
    loaded = GameRecord.load(path)

    assert loaded.game_id == recorder.record.game_id
    assert loaded.player_names == ["Alice", "Bob"]
    assert len(loaded.turns) == 1


def test_round_trip_preserves_move_tile(tmp_path):
    recorder = GameRecorder()
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    recorder.record_turn(game, move)

    path = tmp_path / "game.json"
    recorder.save(path)
    loaded = GameRecord.load(path)

    assert loaded.turns[0].move_tile == move.tile.name


def test_round_trip_preserves_analysis(tmp_path):
    recorder = GameRecorder()
    game = Game()
    game.setup_round()
    move = _first_legal_move(game)
    recorder.record_turn(game, move, analysis={"value_estimate": 0.42})

    path = tmp_path / "game.json"
    recorder.save(path)
    loaded = GameRecord.load(path)

    assert loaded.turns[0].analysis == {"value_estimate": 0.42}
