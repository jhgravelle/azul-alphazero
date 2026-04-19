# tests/test_inspector_endpoint.py

"""Tests for the /inspect endpoints in api/main.py.

Tests the state management (fresh tree vs reconnect, extend) without
exercising the SSE stream itself — that is covered by manual integration
testing.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def _get_a_game_id() -> str:
    """Return a real game_id from the recordings directory, or skip."""
    response = client.get("/recordings")
    assert response.status_code == 200
    recordings = response.json()
    if not recordings:
        pytest.skip("no recordings available — run a game first")
    return recordings[0]["game_id"]


# ── /inspect/{game_id}/{move_index} ──────────────────────────────────────────


def test_inspect_invalid_game_id_returns_404():
    response = client.get("/inspect/no-such-game/0")
    assert response.status_code == 404


def test_inspect_invalid_move_index_returns_422():
    game_id = _get_a_game_id()
    response = client.get(f"/inspect/{game_id}/99999/state")
    assert response.status_code == 422


def test_inspect_state_returns_200():
    """Snapshot endpoint returns current tree state without streaming."""
    game_id = _get_a_game_id()
    response = client.get(f"/inspect/{game_id}/0/state")
    assert response.status_code == 200


def test_inspect_state_has_required_fields():
    game_id = _get_a_game_id()
    response = client.get(f"/inspect/{game_id}/0/state")
    data = response.json()
    for field in ("game_id", "move_index", "sim_count", "done", "tree", "checkpoint"):
        assert field in data, f"missing field: {field!r}"


def test_inspect_state_tree_is_dict():
    game_id = _get_a_game_id()
    response = client.get(f"/inspect/{game_id}/0/state")
    assert isinstance(response.json()["tree"], dict)


def test_inspect_state_sim_count_is_int():
    game_id = _get_a_game_id()
    response = client.get(f"/inspect/{game_id}/0/state")
    assert isinstance(response.json()["sim_count"], int)


def test_inspect_state_done_is_bool():
    game_id = _get_a_game_id()
    response = client.get(f"/inspect/{game_id}/0/state")
    assert isinstance(response.json()["done"], bool)


def test_inspect_state_checkpoint_is_string():
    game_id = _get_a_game_id()
    response = client.get(f"/inspect/{game_id}/0/state")
    assert isinstance(response.json()["checkpoint"], str)


def test_inspect_same_position_reuses_tree():
    """Requesting the same game_id + move_index twice reuses the existing tree."""
    game_id = _get_a_game_id()
    r1 = client.get(f"/inspect/{game_id}/0/state")
    r2 = client.get(f"/inspect/{game_id}/0/state")
    # sim_count should be the same — no extra sims ran
    assert r1.json()["sim_count"] == r2.json()["sim_count"]


def test_inspect_different_move_index_creates_fresh_tree():
    """A different move_index on the same game creates a new tree."""
    game_id = _get_a_game_id()
    r1 = client.get(f"/inspect/{game_id}/0/state")
    r2 = client.get(f"/inspect/{game_id}/1/state")
    # Different positions — tree keys at root will differ
    assert r1.json()["tree"]["key"] != r2.json()["tree"]["key"]


# ── /inspect/extend ───────────────────────────────────────────────────────────


def test_extend_with_no_active_tree_returns_404():
    """Extending before any inspect call returns 404."""
    # Reset inspector state by requesting a fresh position first,
    # then clear it via the reset endpoint.
    client.post("/inspect/reset")
    response = client.post("/inspect/extend")
    assert response.status_code == 404


def test_extend_increases_sim_count():
    game_id = _get_a_game_id()
    client.get(f"/inspect/{game_id}/0/state")
    r1 = client.get(f"/inspect/{game_id}/0/state")
    client.post("/inspect/extend")
    r2 = client.get(f"/inspect/{game_id}/0/state")
    assert r2.json()["sim_count"] > r1.json()["sim_count"]


def test_extend_done_tree_is_still_done():
    """Extending a fully explored tree keeps done=True."""
    game_id = _get_a_game_id()
    # Run until done (may be slow — marked slow)
    for _ in range(20):
        client.post("/inspect/extend")
        state = client.get(f"/inspect/{game_id}/0/state").json()
        if state["done"]:
            break
    if state["done"]:
        client.post("/inspect/extend")
        state2 = client.get(f"/inspect/{game_id}/0/state").json()
        assert state2["done"] is True
    else:
        pytest.skip("tree not fully explored after 20 extends")


# ── /inspect/reset ────────────────────────────────────────────────────────────


def test_reset_clears_inspector_state():
    game_id = _get_a_game_id()
    client.get(f"/inspect/{game_id}/0/state")
    client.post("/inspect/reset")
    response = client.post("/inspect/extend")
    assert response.status_code == 404
