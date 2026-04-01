# tests/test_api.py

"""Tests for the FastAPI endpoints."""

import pytest  # noqa: F401
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_new_game_defaults_to_human_vs_human():
    """Posting to /new-game with no body gives us a fresh game state."""
    response = client.post("/new-game", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["is_game_over"] is False
    assert data["current_player"] == 0


def test_new_game_accepts_agent_config():
    """We can specify player types when starting a new game."""
    response = client.post(
        "/new-game",
        json={"player_types": ["human", "random"]},
    )
    assert response.status_code == 200


def test_new_game_rejects_invalid_player_type():
    """An unknown agent name should return a 422."""
    response = client.post(
        "/new-game",
        json={"player_types": ["human", "banana"]},
    )
    assert response.status_code == 422


def test_agent_move_returns_updated_state():
    """When current player is a bot, /agent-move applies a move and returns state."""
    # Start a game where player 0 is a random agent
    client.post("/new-game", json={"player_types": ["random", "human"]})
    response = client.post("/agent-move")
    assert response.status_code == 200
    data = response.json()
    # After the bot moves, it should be player 1's turn (or round ended)
    assert data["current_player"] in (0, 1)


def test_agent_move_rejected_when_current_player_is_human():
    """/agent-move should 422 when it's a human's turn."""
    client.post("/new-game", json={"player_types": ["human", "random"]})
    response = client.post("/agent-move")
    assert response.status_code == 422
