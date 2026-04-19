# tests/test_inspector_live.py
"""Tests for POST /inspect/live and the shared _game_from_snapshot helper.

Verifies that the live inspector endpoint accepts a game snapshot,
roots an inspector tree there, and reuses the same state machinery
as the recording-based inspector.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fresh_game_snapshot() -> dict:
    """Start a new game and return a snapshot of its initial state."""
    client.post(
        "/new-game",
        json={"player_types": ["human", "human"], "manual_factories": False},
    )
    state = client.get("/state").json()
    return {
        "current_player": state["current_player"],
        "factories": state["factories"],
        "center": state["center"],
        "boards": [
            {
                "score": b["score"],
                "wall": b["wall"],
                "pattern_lines": b["pattern_lines"],
                "floor_line": b["floor_line"],
            }
            for b in state["boards"]
        ],
    }


def _snapshot_after_one_move() -> dict:
    """Return a snapshot after one legal move has been played."""
    client.post(
        "/new-game",
        json={"player_types": ["human", "human"], "manual_factories": False},
    )
    state = client.get("/state").json()
    move = state["legal_moves"][0]
    client.post("/move", json=move)
    state = client.get("/state").json()
    return {
        "current_player": state["current_player"],
        "factories": state["factories"],
        "center": state["center"],
        "boards": [
            {
                "score": b["score"],
                "wall": b["wall"],
                "pattern_lines": b["pattern_lines"],
                "floor_line": b["floor_line"],
            }
            for b in state["boards"]
        ],
    }


# ── POST /inspect/live ────────────────────────────────────────────────────────


def test_inspect_live_returns_200():
    snapshot = _fresh_game_snapshot()
    response = client.post("/inspect/live", json=snapshot)
    assert response.status_code == 200


def test_inspect_live_response_has_required_fields():
    snapshot = _fresh_game_snapshot()
    data = client.post("/inspect/live", json=snapshot).json()
    for field in ("game_id", "move_index", "sim_count", "done", "tree", "checkpoint"):
        assert field in data, f"missing field: {field!r}"


def test_inspect_live_game_id_is_live():
    snapshot = _fresh_game_snapshot()
    data = client.post("/inspect/live", json=snapshot).json()
    assert data["game_id"] == "live"


def test_inspect_live_move_index_is_zero():
    snapshot = _fresh_game_snapshot()
    data = client.post("/inspect/live", json=snapshot).json()
    assert data["move_index"] == 0


def test_inspect_live_tree_is_dict():
    snapshot = _fresh_game_snapshot()
    data = client.post("/inspect/live", json=snapshot).json()
    assert isinstance(data["tree"], dict)


def test_inspect_live_sim_count_is_positive():
    """One batch should have run by the time the response is returned."""
    snapshot = _fresh_game_snapshot()
    data = client.post("/inspect/live", json=snapshot).json()
    assert data["sim_count"] > 0


def test_inspect_live_sets_inspector_state():
    """After /inspect/live, /inspect/extend should work (tree is active)."""
    snapshot = _fresh_game_snapshot()
    client.post("/inspect/live", json=snapshot)
    response = client.post("/inspect/extend")
    assert response.status_code == 200


def test_inspect_live_extend_increases_sim_count():
    snapshot = _fresh_game_snapshot()
    r1 = client.post("/inspect/live", json=snapshot).json()
    r2 = client.post("/inspect/extend").json()
    assert r2["sim_count"] > r1["sim_count"]


def test_inspect_live_different_snapshots_produce_different_trees():
    """Two different game positions should produce trees with different root keys."""
    snap1 = _fresh_game_snapshot()
    r1 = client.post("/inspect/live", json=snap1).json()

    snap2 = _snapshot_after_one_move()
    r2 = client.post("/inspect/live", json=snap2).json()

    assert r1["tree"]["key"] != r2["tree"]["key"]


def test_inspect_live_replaces_existing_inspector_tree():
    """Calling /inspect/live twice replaces the first tree."""
    snap1 = _fresh_game_snapshot()
    client.post("/inspect/live", json=snap1)
    sims_after_first = client.post("/inspect/extend").json()["sim_count"]

    snap2 = _snapshot_after_one_move()
    r2 = client.post("/inspect/live", json=snap2).json()

    # sim_count resets for the new position
    assert r2["sim_count"] <= sims_after_first


def test_inspect_live_state_endpoint_reflects_live_tree():
    """/inspect/live/state returns the same tree as the POST response."""
    snapshot = _fresh_game_snapshot()
    post_data = client.post("/inspect/live", json=snapshot).json()
    get_data = client.get("/inspect/live/state").json()
    assert get_data["game_id"] == "live"
    assert get_data["tree"]["key"] == post_data["tree"]["key"]


# ── Shared machinery reuse ────────────────────────────────────────────────────


def test_inspect_live_then_recording_resets_game_id():
    """Switching from live to a recording-based inspect clears the live state."""
    snapshot = _fresh_game_snapshot()
    client.post("/inspect/live", json=snapshot)

    recordings = client.get("/recordings").json()
    if not recordings:
        pytest.skip("no recordings available")

    game_id = recordings[0]["game_id"]
    data = client.get(f"/inspect/{game_id}/0/state").json()
    assert data["game_id"] == game_id


def test_inspect_reset_clears_live_tree():
    snapshot = _fresh_game_snapshot()
    client.post("/inspect/live", json=snapshot)
    client.post("/inspect/reset")
    response = client.post("/inspect/extend")
    assert response.status_code == 404
