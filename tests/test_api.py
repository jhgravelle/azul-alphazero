# tests/test_api.py

"""Tests for the FastAPI endpoints."""

import pytest
import json
from engine.constants import COL_FOR_TILE_ROW, TILE_FOR_ROW_COL
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
    main._history.clear()
    main._hyp_marker = None
    main._hyp_player_types = None
    main._hyp_agents = None
    main._in_factory_setup = False
    main._factory_cursor = 0
    main._manual_factories = False
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
    game.players[0].score = 10
    game.players[1].score = 10
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


# region GET /recordings ---------------------------------------------------


def test_list_recordings_returns_empty_list_when_no_games_saved(
    client_with_recordings,
):
    client, _ = client_with_recordings
    response = client.get("/recordings")
    assert response.status_code == 200
    assert response.json() == []


def test_list_recordings_returns_entry_after_game_saved(client_with_recordings):
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


# endregion

# region GET /recordings/{game_id} ----------------------------------------


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


# endregion

# region Recording during play --------------------------------------------


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
    total_moves = sum(len(r.moves) for r in main._recorder.record.rounds)
    assert total_moves == 1


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
    total_moves = sum(len(r.moves) for r in main._recorder.record.rounds)
    assert total_moves == 3


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
    assert "rounds" in data


# endregion

# region POST /undo --------------------------------------------------------


def test_undo_after_one_move_restores_previous_state(client):
    """After one move, POST /undo should restore the state to before that move."""
    from api import main

    before = client.get("/state").json()

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    assert client.get("/state").json() != before

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
    assert client.post("/undo").status_code == 400


def test_undo_with_no_history_returns_400(client):
    response = client.post("/undo")
    assert response.status_code == 400


def test_undo_unavailable_in_bot_vs_bot_game(client):
    from api import main

    main._player_types = ["random", "random"]
    client.post("/agent-move")

    response = client.post("/undo")
    assert response.status_code == 400


def test_new_game_clears_undo_history(client):
    from api import main

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    client.post("/new-game", json={})
    assert client.post("/undo").status_code == 400


def test_undo_in_human_vs_bot_skips_back_to_human_turn(client):
    from api import main

    main._player_types = ["human", "random"]
    main._agents = [None, main._make_agent("random")]

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )
    client.post("/agent-move")

    assert main._game.current_player_index == 0

    response = client.post("/undo")
    assert response.status_code == 200
    assert response.json()["current_player"] == 0


# endregion

# region POST /hypothetical/enter ------------------------------------------


def test_enter_hypothetical_returns_200(client):
    response = client.post("/hypothetical/enter")
    assert response.status_code == 200


def test_enter_hypothetical_sets_in_hypothetical_flag(client):
    response = client.post("/hypothetical/enter")
    assert response.json()["in_hypothetical"] is True


def test_enter_hypothetical_overrides_player_types_to_human(client):
    from api import main

    main._player_types = ["human", "greedy"]
    main._agents = [None, main._make_agent("greedy")]

    client.post("/hypothetical/enter")
    assert client.get("/state").json()["player_types"] == ["human", "human"]


def test_enter_hypothetical_cannot_be_entered_twice(client):
    client.post("/hypothetical/enter")
    response = client.post("/hypothetical/enter")
    assert response.status_code == 400


# endregion

# region POST /hypothetical/discard ----------------------------------------


def test_discard_hypothetical_returns_200(client):
    client.post("/hypothetical/enter")
    response = client.post("/hypothetical/discard")
    assert response.status_code == 200


def test_discard_hypothetical_clears_in_hypothetical_flag(client):
    client.post("/hypothetical/enter")
    response = client.post("/hypothetical/discard")
    assert response.json()["in_hypothetical"] is False


def test_discard_hypothetical_restores_state_before_enter(client):
    before = client.get("/state").json()
    client.post("/hypothetical/enter")

    from api import main

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    client.post("/hypothetical/discard")
    assert client.get("/state").json() == before


def test_discard_hypothetical_restores_original_player_types(client):
    from api import main

    main._player_types = ["human", "greedy"]
    main._agents = [None, main._make_agent("greedy")]

    client.post("/hypothetical/enter")
    client.post("/hypothetical/discard")

    assert client.get("/state").json()["player_types"] == ["human", "greedy"]


def test_discard_hypothetical_without_entering_returns_400(client):
    response = client.post("/hypothetical/discard")
    assert response.status_code == 400


# endregion

# region POST /hypothetical/commit -----------------------------------------


def test_commit_hypothetical_returns_200(client):
    client.post("/hypothetical/enter")
    response = client.post("/hypothetical/commit")
    assert response.status_code == 200


def test_commit_hypothetical_clears_in_hypothetical_flag(client):
    client.post("/hypothetical/enter")
    response = client.post("/hypothetical/commit")
    assert response.json()["in_hypothetical"] is False


def test_commit_hypothetical_keeps_moves_made_during_hypothetical(client):
    from api import main

    before = client.get("/state").json()
    client.post("/hypothetical/enter")

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    after_move = client.get("/state").json()
    client.post("/hypothetical/commit")

    assert client.get("/state").json() != before
    assert client.get("/state").json()["current_player"] == after_move["current_player"]


def test_commit_hypothetical_restores_original_player_types(client):
    from api import main

    main._player_types = ["human", "greedy"]
    main._agents = [None, main._make_agent("greedy")]

    client.post("/hypothetical/enter")
    client.post("/hypothetical/commit")

    assert client.get("/state").json()["player_types"] == ["human", "greedy"]


def test_commit_hypothetical_without_entering_returns_400(client):
    response = client.post("/hypothetical/commit")
    assert response.status_code == 400


# endregion

# region Undo interacts correctly with hypothetical mode -------------------


def test_undo_works_inside_hypothetical_mode(client):
    from api import main

    client.post("/hypothetical/enter")
    state_at_enter = client.get("/state").json()

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    client.post("/undo")
    assert client.get("/state").json() == state_at_enter


def test_undo_cannot_go_before_hypothetical_marker(client):
    from api import main

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )

    client.post("/hypothetical/enter")

    move = main._game.legal_moves()[0]
    client.post(
        "/move",
        json={
            "source": move.source,
            "tile": move.tile.name,
            "destination": move.destination,
        },
    )
    client.post("/undo")

    assert client.post("/undo").status_code == 400


# endregion

# region Factory setup: entering setup mode --------------------------------


def test_new_game_with_manual_factories_enters_setup_mode(client):
    response = client.post("/new-game", json={"manual_factories": True})
    assert response.json()["in_factory_setup"] is True


def test_new_game_without_manual_factories_not_in_setup_mode(client):
    response = client.post("/new-game", json={})
    assert response.json()["in_factory_setup"] is False


def test_setup_start_enters_factory_setup_mode(client):
    response = client.post("/setup-factories/start")
    assert response.status_code == 200
    assert response.json()["in_factory_setup"] is True


def test_setup_start_clears_all_factories(client):
    from api import main

    main._game.factories[0] = [main._game.bag.pop() for _ in range(4)]

    client.post("/setup-factories/start")
    state = client.get("/state").json()
    for factory in state["factories"]:
        assert factory == []


def test_setup_start_cursor_at_zero(client):
    response = client.post("/setup-factories/start")
    assert response.json()["factory_cursor"] == 0


def test_factory_cursor_is_none_when_not_in_setup_mode(client):
    response = client.get("/state")
    assert response.json()["factory_cursor"] is None


def test_setup_start_places_first_player_marker_in_center(client):
    client.post("/setup-factories/start")
    state = client.get("/state").json()
    assert "FIRST_PLAYER" in state["center"]


def test_new_game_with_manual_factories_has_first_player_marker_in_center(client):
    response = client.post("/new-game", json={"manual_factories": True})
    assert "FIRST_PLAYER" in response.json()["center"]


# endregion

# region Factory setup: placing tiles -------------------------------------


def test_place_tile_adds_to_first_factory(client):
    from api import main

    client.post("/setup-factories/start")
    color = next(t for t in main._game.bag).name
    client.post("/setup-factories/place", json={"color": color})
    state = client.get("/state").json()
    assert color in state["factories"][0]


def test_place_tile_draws_from_bag(client):
    from api import main

    client.post("/setup-factories/start")
    bag_before = len(main._game.bag)
    color = next(t for t in main._game.bag).name
    client.post("/setup-factories/place", json={"color": color})
    assert len(main._game.bag) == bag_before - 1


def test_place_tile_advances_cursor(client):
    from api import main

    client.post("/setup-factories/start")
    color = next(t for t in main._game.bag).name
    response = client.post("/setup-factories/place", json={"color": color})
    assert response.json()["factory_cursor"] == 1


def test_place_four_tiles_fills_first_factory_and_advances_to_second(client):
    from api import main

    client.post("/setup-factories/start")
    for _ in range(4):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})
    state = client.get("/state").json()
    assert len(state["factories"][0]) == 4
    assert state["factory_cursor"] == 4


def test_place_tile_updates_bag_counts(client):
    from api import main

    client.post("/setup-factories/start")
    color = next(t for t in main._game.bag).name
    bag_before = client.get("/state").json()["bag_counts"][color]
    client.post("/setup-factories/place", json={"color": color})
    bag_after = client.get("/state").json()["bag_counts"][color]
    assert bag_after == bag_before - 1


def test_place_tile_refills_bag_from_discard_when_color_missing(client):
    from api import main
    from engine.constants import Tile

    client.post("/setup-factories/start")
    main._game.bag = [Tile.YELLOW]
    main._game.discard = [Tile.BLUE] * 10

    response = client.post("/setup-factories/place", json={"color": "BLUE"})
    assert response.status_code == 200
    assert "BLUE" in response.json()["factories"][0]


def test_place_tile_returns_400_for_unknown_color(client):
    client.post("/setup-factories/start")
    response = client.post("/setup-factories/place", json={"color": "PURPLE"})
    assert response.status_code == 400


def test_place_tile_returns_400_when_not_in_setup_mode(client):
    from api import main

    color = next(t for t in main._game.bag).name
    response = client.post("/setup-factories/place", json={"color": color})
    assert response.status_code == 400


# endregion

# region Factory setup: removing tiles ------------------------------------


def test_remove_tile_returns_it_to_bag(client):
    from api import main

    client.post("/setup-factories/start")
    color = next(t for t in main._game.bag).name
    client.post("/setup-factories/place", json={"color": color})
    bag_after_place = len(main._game.bag)

    client.post("/setup-factories/remove", json={"factory": 0, "slot": 0})
    assert len(main._game.bag) == bag_after_place + 1


def test_remove_tile_empties_that_slot(client):
    from api import main

    client.post("/setup-factories/start")
    color = next(t for t in main._game.bag).name
    client.post("/setup-factories/place", json={"color": color})

    client.post("/setup-factories/remove", json={"factory": 0, "slot": 0})
    state = client.get("/state").json()
    assert state["factories"][0] == []


def test_remove_tile_sets_cursor_to_removed_slot(client):
    from api import main

    client.post("/setup-factories/start")
    for _ in range(2):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    response = client.post("/setup-factories/remove", json={"factory": 0, "slot": 0})
    assert response.json()["factory_cursor"] == 0


def test_remove_tile_returns_400_for_empty_slot(client):
    client.post("/setup-factories/start")
    response = client.post("/setup-factories/remove", json={"factory": 0, "slot": 0})
    assert response.status_code == 400


def test_remove_tile_returns_400_when_not_in_setup_mode(client):
    response = client.post("/setup-factories/remove", json={"factory": 0, "slot": 0})
    assert response.status_code == 400


# endregion

# region Factory setup: restart -------------------------------------------


def test_restart_returns_all_tiles_to_bag(client):
    from api import main

    client.post("/setup-factories/start")
    bag_at_start = len(main._game.bag)

    for _ in range(4):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    client.post("/setup-factories/restart")
    assert len(main._game.bag) == bag_at_start


def test_restart_clears_all_factories(client):
    from api import main

    client.post("/setup-factories/start")
    for _ in range(4):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    client.post("/setup-factories/restart")
    state = client.get("/state").json()
    for factory in state["factories"]:
        assert factory == []


def test_restart_resets_cursor_to_zero(client):
    from api import main

    client.post("/setup-factories/start")
    for _ in range(4):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    response = client.post("/setup-factories/restart")
    assert response.json()["factory_cursor"] == 0


def test_restart_returns_400_when_not_in_setup_mode(client):
    response = client.post("/setup-factories/restart")
    assert response.status_code == 400


# endregion

# region Factory setup: random fill ----------------------------------------


def test_random_fills_all_remaining_placeholders(client):
    client.post("/setup-factories/start")
    client.post("/setup-factories/random")
    state = client.get("/state").json()
    for factory in state["factories"]:
        assert len(factory) == 4


def test_random_fills_but_stays_in_setup_mode(client):
    client.post("/setup-factories/start")
    response = client.post("/setup-factories/random")
    data = response.json()
    assert data["in_factory_setup"] is True
    for factory in data["factories"]:
        assert len(factory) == 4


def test_random_with_some_tiles_already_placed(client):
    from api import main

    client.post("/setup-factories/start")
    for _ in range(2):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    client.post("/setup-factories/random")
    state = client.get("/state").json()
    for factory in state["factories"]:
        assert len(factory) == 4


def test_random_returns_400_when_not_in_setup_mode(client):
    response = client.post("/setup-factories/random")
    assert response.status_code == 400


# endregion

# region Factory setup: commit --------------------------------------------


def test_commit_exits_setup_mode(client):
    from api import main

    client.post("/setup-factories/start")
    num_factories = len(main._game.factories)
    for _ in range(num_factories * 4):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    response = client.post("/setup-factories/commit")
    assert response.status_code == 200
    assert response.json()["in_factory_setup"] is False


def test_commit_returns_400_when_factories_not_full(client):
    client.post("/setup-factories/start")
    from api import main

    for _ in range(3):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    response = client.post("/setup-factories/commit")
    assert response.status_code == 400


def test_commit_returns_400_when_not_in_setup_mode(client):
    response = client.post("/setup-factories/commit")
    assert response.status_code == 400


# endregion

# region Factory setup: new-game integration ------------------------------


def test_manual_factories_setting_persists_on_state(client):
    client.post("/new-game", json={"manual_factories": True})
    assert client.get("/state").json()["manual_factories"] is True


def test_manual_factories_false_by_default(client):
    client.post("/new-game", json={})
    assert client.get("/state").json()["manual_factories"] is False


# endregion

# region Factory setup: tile conservation ---------------------------------


def test_total_tiles_conserved_after_place_and_remove(client):
    from api import main

    def total_tiles():
        g = main._game
        return len(g.bag) + len(g.discard) + sum(len(f) for f in g.factories)

    client.post("/setup-factories/start")
    before = total_tiles()

    for _ in range(4):
        color = next(t for t in main._game.bag).name
        client.post("/setup-factories/place", json={"color": color})

    assert total_tiles() == before

    client.post("/setup-factories/remove", json={"factory": 0, "slot": 0})

    assert total_tiles() == before


def test_place_rejected_when_cursor_is_past_end(client):
    from api import main

    client.post("/setup-factories/start")
    main._factory_cursor = len(main._game.factories) * 4

    color = next(t for t in main._game.bag).name
    response = client.post("/setup-factories/place", json={"color": color})
    assert response.status_code == 400


def test_total_tiles_conserved_after_restart(client):
    from api import main

    def total_tiles():
        g = main._game
        return len(g.bag) + len(g.discard) + sum(len(f) for f in g.factories)

    client.post("/setup-factories/start")
    bag_at_start = len(main._game.bag)
    before = total_tiles()

    client.post("/setup-factories/random")
    assert total_tiles() == before

    client.post("/setup-factories/restart")
    assert total_tiles() == before
    assert len(main._game.bag) == bag_at_start


def test_total_tiles_conserved_after_clicking_placed_tile_when_full(client):
    from api import main

    def total_tiles():
        g = main._game
        return len(g.bag) + len(g.discard) + sum(len(f) for f in g.factories)

    client.post("/setup-factories/start")
    before = total_tiles()

    client.post("/setup-factories/random")
    assert total_tiles() == before

    client.post("/setup-factories/remove", json={"factory": 0, "slot": 0})
    assert total_tiles() == before


# endregion

# region Factory setup: random is truly random after restart ---------------


def test_restart_shuffles_bag(client):
    from api import main

    client.post("/setup-factories/start")
    client.post("/setup-factories/random")

    g = main._game
    tiles_in_factories = [t for f in g.factories for t in f]
    tiles_in_bag = list(g.bag)
    all_tiles_sorted = sorted(t.name for t in tiles_in_factories + tiles_in_bag)
    unshuffled_order = tiles_in_factories + tiles_in_bag

    client.post("/setup-factories/restart")
    bag_after = list(main._game.bag)

    assert sorted(t.name for t in bag_after) == all_tiles_sorted
    assert bag_after != unshuffled_order


def test_random_after_restart_gives_different_factories(client):
    client.post("/setup-factories/start")
    r1 = client.post("/setup-factories/random").json()
    factories_first = [list(f) for f in r1["factories"]]

    client.post("/setup-factories/restart")
    r2 = client.post("/setup-factories/random").json()
    factories_second = [list(f) for f in r2["factories"]]

    assert factories_first != factories_second


def test_random_enabled_when_factories_already_full(client):
    client.post("/setup-factories/start")
    client.post("/setup-factories/random")

    response = client.post("/setup-factories/random")
    assert response.status_code == 200
    for factory in response.json()["factories"]:
        assert len(factory) == 4


def test_random_when_full_reshuffles_and_refills(client):
    client.post("/setup-factories/start")
    client.post("/setup-factories/random")

    response = client.post("/setup-factories/random")
    assert response.status_code == 200
    data = response.json()
    assert data["in_factory_setup"] is True
    for factory in data["factories"]:
        assert len(factory) == 4


# endregion

# region POST /hypothetical/from-snapshot ----------------------------------


def test_from_snapshot_enters_hypothetical_mode(client):
    from api import main

    snapshot = _make_snapshot(main._game)
    response = client.post("/hypothetical/from-snapshot", json=snapshot)
    assert response.status_code == 200
    assert response.json()["in_hypothetical"] is True


def test_from_snapshot_loads_factories_and_center(client):
    from api import main
    from engine.constants import Tile

    main._game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    main._game.center = [Tile.FIRST_PLAYER]

    snapshot = _make_snapshot(main._game)
    response = client.post("/hypothetical/from-snapshot", json=snapshot)
    data = response.json()

    assert data["factories"][0] == ["BLUE", "BLUE", "RED", "YELLOW"]
    assert "FIRST_PLAYER" in data["center"]


def test_from_snapshot_loads_board_states(client):
    from api import main

    main._game.players[0].score = 7
    main._game.players[1].score = 3

    snapshot = _make_snapshot(main._game)
    response = client.post("/hypothetical/from-snapshot", json=snapshot)
    data = response.json()

    assert data["boards"][0]["score"] == 7
    assert data["boards"][1]["score"] == 3


def test_from_snapshot_overrides_player_types_to_human(client):
    from api import main

    main._player_types = ["human", "greedy"]
    main._agents = [None, main._make_agent("greedy")]

    snapshot = _make_snapshot(main._game)
    response = client.post("/hypothetical/from-snapshot", json=snapshot)
    assert response.json()["player_types"] == ["human", "human"]


def test_from_snapshot_rejected_when_already_in_hypothetical(client):
    from api import main

    client.post("/hypothetical/enter")
    snapshot = _make_snapshot(main._game)
    response = client.post("/hypothetical/from-snapshot", json=snapshot)
    assert response.status_code == 400


def test_from_snapshot_legal_moves_are_valid(client):
    from api import main

    snapshot = _make_snapshot(main._game)
    response = client.post("/hypothetical/from-snapshot", json=snapshot)
    assert len(response.json()["legal_moves"]) > 0


def test_discard_after_from_snapshot_restores_original_game(client):
    from api import main
    from engine.constants import Tile
    import copy

    before = client.get("/state").json()

    modified = copy.deepcopy(main._game)
    modified.factories[0] = [Tile.RED, Tile.RED, Tile.RED, Tile.RED]
    snapshot = _make_snapshot(modified)

    client.post("/hypothetical/from-snapshot", json=snapshot)
    client.post("/hypothetical/discard")
    assert client.get("/state").json() == before


# endregion

# region Round and manual factory integration ------------------------------


def test_round_ends_and_api_starts_next_round_automatically(client):
    from api import main

    round_before = main._game.round

    for _ in range(500):
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
        state = client.get("/state").json()
        if state["round"] > round_before:
            break

    state = client.get("/state").json()
    if not state["is_game_over"]:
        assert state["round"] == round_before + 1
        total_tiles = sum(len(f) for f in state["factories"])
        assert total_tiles > 0


def test_manual_factories_persists_to_second_round(client):
    from api import main

    client.post("/new-game", json={"manual_factories": True})

    client.post("/setup-factories/random")
    client.post("/setup-factories/commit")

    for _ in range(500):
        state = client.get("/state").json()
        if state["in_factory_setup"] or state["is_game_over"]:
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

    state = client.get("/state").json()
    if not state["is_game_over"]:
        assert state["in_factory_setup"] is True
        for factory in state["factories"]:
            assert factory == []


def test_entering_factory_setup_refills_bag_when_empty(client):
    from api import main

    client.post(
        "/new-game",
        json={"player_types": ["human", "human"], "manual_factories": True},
    )

    game = main._game
    game.discard.extend(game.bag)
    game.bag.clear()
    assert len(game.bag) == 0
    assert len(game.discard) > 0

    main._enter_factory_setup()

    response = client.get("/state")
    body = response.json()
    bag_total = sum(body["bag_counts"].values())
    assert bag_total > 0
    discard_total = sum(body["discard_counts"].values())
    assert discard_total == 0


# endregion

# region Helper ------------------------------------------------------------


def _encode_pattern_lines_for_snapshot(player) -> list[list[str]]:
    """Serialize pattern_grid into the pattern_lines wire format for snapshot tests."""
    result = []
    for row in range(5):
        tile = player._line_tile(row)
        if tile is None:
            result.append([])
        else:
            col = COL_FOR_TILE_ROW[tile][row]
            count = player.pattern_grid[row][col]
            result.append([tile.name] * count)
    return result


def _make_snapshot(game) -> dict:
    """Build a minimal from-snapshot payload from a live Game object."""
    return {
        "factories": [[t.name for t in f] for f in game.factories],
        "center": [t.name for t in game.center],
        "boards": [
            {
                "score": p.score,
                "wall": [
                    [
                        TILE_FOR_ROW_COL[row][col].name if p.wall[row][col] else None
                        for col in range(5)
                    ]
                    for row in range(5)
                ],
                "pattern_lines": _encode_pattern_lines_for_snapshot(p),
                "floor_line": [t.name for t in p.floor_line],
            }
            for p in game.players
        ],
    }


# endregion
