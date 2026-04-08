# tests/test_api.py

"""Tests for the FastAPI endpoints."""

import pytest
import json
from engine.game import Game
from fastapi.testclient import TestClient

# from pathlib import Path


@pytest.fixture
def client():
    from api import main
    from api.main import app

    main._game = Game()
    main._game.setup_round()
    main._player_types = ["human", "human"]
    main._agents = [None, None]
    main._history.clear()
    return TestClient(app)


def test_new_game_defaults_to_human_vs_human(client):
    response = client.post("/new-game", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["is_game_over"] is False
    assert data["current_player"] == 0


def test_new_game_accepts_agent_config(client):
    response = client.post(
        "/new-game",
        json={"player_types": ["human", "random"]},
    )
    assert response.status_code == 200


def test_new_game_rejects_invalid_player_type(client):
    response = client.post(
        "/new-game",
        json={"player_types": ["human", "banana"]},
    )
    assert response.status_code == 422


def test_agent_move_returns_updated_state(client):
    client.post("/new-game", json={"player_types": ["random", "human"]})
    response = client.post("/agent-move")
    assert response.status_code == 200
    data = response.json()
    assert data["current_player"] in (0, 1)


def test_agent_move_rejected_when_current_player_is_human(client):
    client.post("/new-game", json={"player_types": ["human", "random"]})
    response = client.post("/agent-move")
    assert response.status_code == 422


def test_tied_game_reports_no_winner(client):
    """If two players finish with the same score, winner should be None."""
    from api.main import _build_response
    from engine.game import Game

    game = Game()
    game.setup_round()
    # Both players start at 0 — a freshly set up game is always tied
    game.state.players[0].score = 10
    game.state.players[1].score = 10
    response = _build_response(game)
    assert response.winner is None


@pytest.fixture
def client_with_recordings(tmp_path, monkeypatch):
    """Client with recordings directory redirected to a temp folder."""
    from api import main
    from api.main import app

    monkeypatch.setattr(main, "_RECORDINGS_DIR", tmp_path)
    main._game = Game()
    main._game.setup_round()
    main._player_types = ["human", "human"]
    main._agents = [None, None]
    main._recorder = None
    return TestClient(app), tmp_path


# ── GET /recordings ────────────────────────────────────────────────────────


def test_list_recordings_returns_empty_list_when_no_games_saved(
    client_with_recordings,
):
    client, _ = client_with_recordings
    response = client.get("/recordings")
    assert response.status_code == 200
    assert response.json() == []


def test_list_recordings_returns_entry_after_game_saved(client_with_recordings):
    client, tmp_path = client_with_recordings
    # Write a fake recording directly into the temp dir
    record = {
        "game_id": "abc-123",
        "timestamp": "2026-04-05T00:00:00+00:00",
        "player_names": ["Alice", "Bob"],
        "turns": [],
        "final_scores": [10, 8],
        "winner": 0,
    }
    (tmp_path / "abc-123.json").write_text(json.dumps(record))
    response = client.get("/recordings")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["game_id"] == "abc-123"
    assert data[0]["player_names"] == ["Alice", "Bob"]
    assert data[0]["final_scores"] == [10, 8]
    assert data[0]["winner"] == 0


def test_list_recordings_returns_multiple_entries(client_with_recordings):
    client, tmp_path = client_with_recordings
    for i in range(3):
        record = {
            "game_id": f"game-{i}",
            "timestamp": "2026-04-05T00:00:00+00:00",
            "player_names": ["Alice", "Bob"],
            "turns": [],
            "final_scores": [10, 8],
            "winner": 0,
        }
        (tmp_path / f"game-{i}.json").write_text(json.dumps(record))
    response = client.get("/recordings")
    assert response.status_code == 200
    assert len(response.json()) == 3


# ── GET /recordings/{game_id} ──────────────────────────────────────────────


def test_get_recording_returns_full_record(client_with_recordings):
    client, tmp_path = client_with_recordings
    record = {
        "game_id": "abc-123",
        "timestamp": "2026-04-05T00:00:00+00:00",
        "player_names": ["Alice", "Bob"],
        "turns": [],
        "final_scores": [10, 8],
        "winner": 0,
    }
    (tmp_path / "abc-123.json").write_text(json.dumps(record))
    response = client.get("/recordings/abc-123")
    assert response.status_code == 200
    data = response.json()
    assert data["game_id"] == "abc-123"
    assert data["player_names"] == ["Alice", "Bob"]


def test_get_recording_returns_404_for_unknown_game_id(client_with_recordings):
    client, _ = client_with_recordings
    response = client.get("/recordings/does-not-exist")
    assert response.status_code == 404


# ── Recording during play ──────────────────────────────────────────────────


def test_new_game_creates_recorder(client_with_recordings):
    client, _ = client_with_recordings
    from api import main

    client.post("/new-game", json={})
    assert main._recorder is not None


def test_move_is_recorded(client_with_recordings):
    client, _ = client_with_recordings
    from api import main

    client.post("/new-game", json={})
    moves = main._game.legal_moves()
    move = moves[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )
    assert main._recorder is not None
    assert len(main._recorder.record.turns) == 1


def test_multiple_moves_are_all_recorded(client_with_recordings):
    client, _ = client_with_recordings
    from api import main

    client.post("/new-game", json={})
    for _ in range(3):
        if main._game.legal_moves():
            move = main._game.legal_moves()[0]
            client.post(
                "/move",
                json={
                    "source": move.source,
                    "tile": move.tile.name,
                    "destination": move.destination,
                },
            )
    assert main._recorder is not None
    assert len(main._recorder.record.turns) == 3


def test_completed_game_saves_recording_to_disk(client_with_recordings):
    """Play moves until the game ends and verify a file was written."""
    client, tmp_path = client_with_recordings
    from api import main

    client.post("/new-game", json={})
    max_moves = 500
    for _ in range(max_moves):
        if main._game.is_game_over():
            break
        moves = main._game.legal_moves()
        if not moves:
            break
        move = moves[0]
        client.post(
            "/move",
            json={
                "source": move.source,
                "tile": move.tile.name,
                "destination": move.destination,
            },
        )

    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 1


def test_saved_recording_is_valid_json(client_with_recordings):
    client, tmp_path = client_with_recordings
    from api import main

    client.post("/new-game", json={})
    max_moves = 500
    for _ in range(max_moves):
        if main._game.is_game_over():
            break
        moves = main._game.legal_moves()
        if not moves:
            break
        move = moves[0]
        client.post(
            "/move",
            json={
                "source": move.source,
                "tile": move.tile.name,
                "destination": move.destination,
            },
        )

    saved_files = list(tmp_path.glob("*.json"))
    assert len(saved_files) == 1
    data = json.loads(saved_files[0].read_text())
    assert "game_id" in data
    assert "turns" in data


# ── POST /undo ─────────────────────────────────────────────────────────────


def test_undo_after_one_move_restores_previous_state(client):
    """After one move, POST /undo should restore the state to before that move."""
    from api import main

    # Capture the full state response before any move.
    before = client.get("/state").json()

    # Make exactly one legal move.
    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    # Confirm the state changed.
    assert client.get("/state").json() != before

    # Undo — state should be identical to what it was before the move.
    response = client.post("/undo")
    assert response.status_code == 200
    assert client.get("/state").json() == before


def test_undo_decrements_history_each_time(client):
    """Each undo pops exactly one entry; a third undo on two moves returns 400."""
    from api import main

    for _ in range(2):
        move = main._game.legal_moves()[0]
        client.post(
            "/move",
            json={
                "source": move.source,
                "tile": move.tile.name,
                "destination": move.destination,
            },
        )

    assert client.post("/undo").status_code == 200
    assert client.post("/undo").status_code == 200
    # History is now empty — third undo must fail.
    assert client.post("/undo").status_code == 400


def test_undo_with_no_history_returns_400(client):
    """POST /undo on a fresh game with no moves made should return 400."""
    response = client.post("/undo")
    assert response.status_code == 400


def test_undo_unavailable_in_bot_vs_bot_game(client):
    """Undo is disabled when both players are agents — no human to undo for."""
    from api import main

    # Switch both players to agents.
    main._player_types = ["random", "random"]

    # Make an agent move so history would exist if undo were permitted.
    client.post("/agent-move")

    response = client.post("/undo")
    assert response.status_code == 400


def test_new_game_clears_undo_history(client):
    """Starting a new game must clear any existing undo history."""
    from api import main

    # Make one move to populate history.
    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    # Start a fresh game.
    client.post("/new-game", json={})

    # Undo should now fail — history was cleared by /new-game.
    assert client.post("/undo").status_code == 400
