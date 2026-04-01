# tests/test_api.py

"""Tests for the FastAPI endpoints."""

import pytest
from engine.game import Game
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api import main
    from api.main import app

    main._game = Game()
    main._game.setup_round()
    main._player_types = ["human", "human"]
    main._agents = [None, None]
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
