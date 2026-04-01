# tests/test_api.py

"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """A fresh TestClient (and reset global game) for each test."""
    from api import main
    from api.main import app

    main._game = None
    main._player_types = None
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
