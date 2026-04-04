# tests/test_api_scoring.py

"""Tests that BoardResponse includes pending scoring breakdown fields."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _fresh_boards() -> list[dict]:
    response = client.post("/new-game", json={"player_types": ["human", "human"]})
    assert response.status_code == 200
    return response.json()["boards"]


# ── Field presence ─────────────────────────────────────────────────────────────


def test_board_response_has_pending_placements():
    for board in _fresh_boards():
        assert "pending_placements" in board


def test_board_response_has_pending_bonuses():
    for board in _fresh_boards():
        assert "pending_bonuses" in board


# ── Values at game start ───────────────────────────────────────────────────────


def test_pending_placements_is_empty_at_game_start():
    for board in _fresh_boards():
        assert board["pending_placements"] == []


def test_pending_bonuses_is_empty_at_game_start():
    for board in _fresh_boards():
        assert board["pending_bonuses"] == []


# ── Shape of PendingPlacement ──────────────────────────────────────────────────


def test_pending_placement_has_required_fields():
    """Each PendingPlacement must have row, column, and placement_points."""
    for board in _fresh_boards():
        for placement in board["pending_placements"]:
            assert "row" in placement
            assert "column" in placement
            assert "placement_points" in placement


def test_pending_placement_points_is_int():
    for board in _fresh_boards():
        for placement in board["pending_placements"]:
            assert isinstance(placement["placement_points"], int)


# ── Shape of PendingBonus ──────────────────────────────────────────────────────


def test_pending_bonus_has_required_fields():
    """Each PendingBonus must have bonus_type, index, and bonus_points."""
    for board in _fresh_boards():
        for bonus in board["pending_bonuses"]:
            assert "bonus_type" in bonus
            assert "index" in bonus
            assert "bonus_points" in bonus


def test_pending_bonus_type_is_valid():
    for board in _fresh_boards():
        for bonus in board["pending_bonuses"]:
            assert bonus["bonus_type"] in ("row", "column", "tile")


def test_pending_bonus_points_is_positive_int():
    for board in _fresh_boards():
        for bonus in board["pending_bonuses"]:
            assert isinstance(bonus["bonus_points"], int)
            assert bonus["bonus_points"] > 0


# ── Invariants ─────────────────────────────────────────────────────────────────


def test_pending_placement_row_is_in_range():
    for board in _fresh_boards():
        for placement in board["pending_placements"]:
            assert 0 <= placement["row"] <= 4


def test_pending_placement_column_is_in_range():
    for board in _fresh_boards():
        for placement in board["pending_placements"]:
            assert 0 <= placement["column"] <= 4


def test_pending_bonus_index_is_non_negative():
    for board in _fresh_boards():
        for bonus in board["pending_bonuses"]:
            assert bonus["index"] >= 0
