# api/main.py
"""FastAPI application for the Azul game."""

from datetime import datetime
import json
import logging
import random
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.base import Agent
from agents.alphabeta import AlphaBetaAgent
from agents.alphazero import AlphaZeroAgent
from agents.registry import make_agent, AGENT_REGISTRY
from api.schemas import (
    BoardResponse,
    GameStateResponse,
    HypotheticalSnapshotRequest,
    MoveRequest,
    NewGameRequest,
    PendingBonus,
    PendingPlacement,
    PlaceTileRequest,
    PlayerType,
    RecordingSummary,
    RemoveTileRequest,
)
from engine.constants import COL_FOR_TILE_ROW, TILE_FOR_ROW_COL, Tile
from engine.game import Game, Move
from engine.game_recorder import (
    GameRecorder,
    GameRecord,
    _pending_placement_details,
    _pending_bonus_details,
    _build_post_placement_wall,
)
from engine.player import Player
from engine.replay import replay_to_move
from neural.search_tree import PolicyValueFn, SearchTree, make_policy_value_fn

logger = logging.getLogger(__name__)

app = FastAPI(title="Azul API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_RECORDINGS_DIR = Path("recordings")

_game = Game()
_game.setup_round()
_player_types: list[PlayerType] = ["human", "human"]
_agents: list[Agent | None] = [None, None]
_recorder: GameRecorder | None = None
_history: list[Game] = []

_hyp_marker: int | None = None
_hyp_player_types: list[PlayerType] | None = None
_hyp_agents: list[Agent | None] | None = None

_manual_factories: bool = False
_in_factory_setup: bool = False
_factory_cursor: int = 0
_last_game_id: str | None = None
_search_tree: SearchTree | None = None

_inspector_tree: SearchTree | None = None
_inspector_game_id: str | None = None
_inspector_move_index: int | None = None
_inspector_sim_count: int = 0
_inspector_snapshot_agent: str = "minimax"
_inspector_snapshot_sims: int = 200
_INSPECTOR_BATCH = 1000


def _make_agent(player_type: PlayerType) -> Agent | None:
    return make_agent(player_type)


# region Pattern line serialization ----------------------------------------


def _encode_pattern_lines(player: Player) -> list[list[str]]:
    """Serialize pattern_grid into the pattern_lines wire format.

    Each row is a list of tile name strings — only the committed color tile
    appears, repeated by fill count. Empty rows are empty lists.
    """
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


def _decode_pattern_lines(player: Player, pattern_lines: list[list[str]]) -> None:
    """Decode the pattern_lines wire format back into player.pattern_grid.

    Each row is a list of tile name strings. Fill count equals list length.
    Clears all grid cells first, then sets the appropriate cell for each row.
    """
    for row in range(5):
        for col in range(5):
            player.pattern_grid[row][col] = 0
    for row, tiles in enumerate(pattern_lines):
        if not tiles:
            continue
        tile = _str_to_tile(tiles[0])
        col = COL_FOR_TILE_ROW[tile][row]
        player.pattern_grid[row][col] = len(tiles)


def _encode_wall(player: Player) -> list[list[str | None]]:
    """Serialize the binary wall grid into tile name strings.

    Filled cells carry the wall pattern color for that position; empty cells
    are None.
    """
    return [
        [
            TILE_FOR_ROW_COL[row][col].name if player.wall[row][col] else None
            for col in range(5)
        ]
        for row in range(5)
    ]


def _decode_wall(wall_data: list[list[str | None]]) -> list[list[int]]:
    """Decode the wall wire format into a binary 5×5 grid."""
    return [[1 if cell is not None else 0 for cell in row] for row in wall_data]


# endregion


def _game_from_snapshot(request: HypotheticalSnapshotRequest) -> Game:
    """Reconstruct a bare Game object from a hypothetical snapshot request."""
    game = Game()
    game.current_player_index = request.current_player

    for factory, tile_names in zip(game.factories, request.factories):
        factory.clear()
        for name in tile_names:
            factory.append(_str_to_tile(name))

    game.center.clear()
    for name in request.center:
        game.center.append(_str_to_tile(name))

    for player, board_req in zip(game.players, request.boards):
        player.score = board_req.score
        _decode_pattern_lines(player, board_req.pattern_lines)
        player.wall = _decode_wall(board_req.wall)
        player.floor_line = [_str_to_tile(name) for name in board_req.floor_line]
        player._update_pending()
        player._update_penalty()
        player._update_bonus()

    return game


def _uniform_pv(game: Game, legal: list[Move]) -> tuple[list[float], float]:
    n = len(legal)
    return ([1.0 / n] * n if n else []), 0.0


def _tile_to_str(tile: Tile) -> str:
    return tile.name


def _str_to_tile(name: str) -> Tile:
    try:
        return Tile[name]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown tile color: {name!r}")


def _str_to_setup_tile(name: str) -> Tile:
    try:
        tile = Tile[name]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown tile color: {name!r}")
    if tile == Tile.FIRST_PLAYER:
        raise HTTPException(
            status_code=400, detail="Cannot place the first-player marker in a factory"
        )
    return tile


def _push_history() -> None:
    """Push a deep copy of the current game onto the undo stack."""
    _history.append(_game.clone())


def _build_pending(
    player: Player,
) -> tuple[list[PendingPlacement], list[PendingBonus]]:
    """Compute pending placements and bonuses for a player without mutating it."""
    post_wall = _build_post_placement_wall(player)
    placement_details = _pending_placement_details(player)
    bonus_details = _pending_bonus_details(post_wall)

    placements = [
        PendingPlacement(
            row=d["row"], column=d["column"], placement_points=d["placement_points"]
        )
        for d in placement_details
    ]
    bonuses = [
        PendingBonus(
            bonus_type=d["bonus_type"],
            index=d["index"],
            bonus_points=d["bonus_points"],
        )
        for d in bonus_details
    ]
    return placements, bonuses


def _build_response(game: Game) -> GameStateResponse:
    """Translate the engine Game object into a GameStateResponse."""
    boards = []
    for player in game.players:
        pattern_lines = _encode_pattern_lines(player)
        wall = _encode_wall(player)
        floor_line = [_tile_to_str(t) for t in player.floor_line]
        pending_placements, pending_bonuses = _build_pending(player)
        boards.append(
            BoardResponse(
                score=player.score,
                pattern_lines=pattern_lines,
                wall=wall,
                floor_line=floor_line,
                pending_placements=pending_placements,
                pending_bonuses=pending_bonuses,
            )
        )

    game_over = game.is_game_over()
    winner = None
    if game_over:
        scores = [p.score for p in game.players]
        if scores.count(max(scores)) == 1:
            winner = scores.index(max(scores))

    legal = [
        MoveRequest(source=m.source, tile=m.tile.name, destination=m.destination)
        for m in game.legal_moves()
    ]

    colors = [t for t in Tile if t != Tile.FIRST_PLAYER]

    def _counts(tile_list: list[Tile]) -> dict[str, int]:
        return {t.name: tile_list.count(t) for t in colors}

    return GameStateResponse(
        current_player=game.current_player_index,
        factories=[[_tile_to_str(t) for t in f] for f in game.factories],
        center=[_tile_to_str(t) for t in game.center],
        boards=boards,
        is_game_over=game_over,
        winner=winner,
        legal_moves=legal,
        player_types=_player_types,
        round=game.round,
        bag_counts=_counts(game.bag),
        discard_counts=_counts(game.discard),
        in_hypothetical=_hyp_marker is not None,
        in_factory_setup=_in_factory_setup,
        factory_cursor=_factory_cursor if _in_factory_setup else None,
        manual_factories=_manual_factories,
        last_game_id=_last_game_id,
    )


def _save_recording(recorder: GameRecorder, game: Game) -> None:
    global _last_game_id
    try:
        recorder.finalize(game)
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.fromisoformat(recorder.record.timestamp)
        date_str = timestamp.strftime("%Y%m%d %H%M%S")
        scores = recorder.record.final_scores
        names = recorder.record.player_names
        players_str = " - ".join(
            f"{name} {score}" for name, score in zip(names, scores)
        )
        filename = f"{date_str} {players_str}.json"
        path = _RECORDINGS_DIR / filename
        recorder.save(path)
        _last_game_id = recorder.record.game_id
        logger.info("saved recording %s", filename)
    except Exception:
        logger.exception("failed to save recording")


def _exit_hypothetical(*, keep_state: bool) -> None:
    global _hyp_marker, _hyp_player_types, _hyp_agents, _player_types, _agents

    marker = _hyp_marker
    saved_types = _hyp_player_types
    saved_agents = _hyp_agents

    assert marker is not None
    assert saved_types is not None
    assert saved_agents is not None

    _hyp_marker = None
    _hyp_player_types = None
    _hyp_agents = None

    _player_types = saved_types
    _agents = saved_agents

    if keep_state:
        del _history[marker:]
    else:
        if marker < len(_history):
            _game.__dict__.update(_history[marker].__dict__)
        del _history[marker:]


def _enter_factory_setup() -> None:
    global _in_factory_setup, _factory_cursor
    for factory in _game.factories:
        factory.clear()
    _game.center.clear()
    _game.center.append(Tile.FIRST_PLAYER)

    if not _game.bag and _game.discard:
        _game.bag.extend(_game.discard)
        _game.discard.clear()
        random.shuffle(_game.bag)

    _in_factory_setup = True
    _factory_cursor = 0


def _handle_round_end() -> None:
    round_ended = _game.advance(skip_setup=_manual_factories)
    if round_ended:
        if _manual_factories:
            _enter_factory_setup()
        if _search_tree is not None:
            _search_tree.reset(_game)
        if _recorder is not None:
            _recorder.start_round(_game)


def _total_slots() -> int:
    return len(_game.factories) * 4


def _draw_one(tile: Tile) -> None:
    if tile not in _game.bag:
        if not _game.discard:
            raise HTTPException(
                status_code=400,
                detail=f"No {tile.name} tiles available in bag or discard",
            )
        _game.bag.extend(_game.discard)
        _game.discard.clear()
        random.shuffle(_game.bag)
        if tile not in _game.bag:
            raise HTTPException(
                status_code=400,
                detail=f"No {tile.name} tiles available in bag or discard",
            )
    _game.bag.remove(tile)


def _flat_to_factory_slot(cursor: int) -> tuple[int, int]:
    return cursor // 4, cursor % 4


def _next_cursor() -> int:
    for i in range(_factory_cursor, _total_slots()):
        factory_index, slot_index = _flat_to_factory_slot(i)
        if slot_index >= len(_game.factories[factory_index]):
            return i
    return _total_slots()


# region Endpoints ---------------------------------------------------------


@app.get("/agents")
def list_agents() -> list[dict]:
    return [
        {"value": name, "label": label}
        for name, label, _, hidden in AGENT_REGISTRY
        if name != "human" and not hidden
    ]


@app.get("/state", response_model=GameStateResponse)
def get_state() -> GameStateResponse:
    return _build_response(_game)


@app.post("/move", response_model=GameStateResponse)
def make_move(move_request: MoveRequest) -> GameStateResponse:
    global _recorder

    tile = _str_to_tile(move_request.tile)
    move = Move(
        source=move_request.source, tile=tile, destination=move_request.destination
    )
    if move not in _game.legal_moves():
        raise HTTPException(status_code=422, detail="Illegal move")

    if _recorder is not None:
        _recorder.record_move(move, player_index=_game.current_player_index)

    _push_history()
    _game.make_move(move)

    if _search_tree is not None:
        _search_tree.advance(move)

    if _hyp_marker is not None:
        _game.advance()
    else:
        _handle_round_end()
        if _game.is_game_over() and _recorder is not None:
            _save_recording(_recorder, _game)
            _recorder = None

    return _build_response(_game)


@app.post("/new-game", response_model=GameStateResponse)
def new_game(request: NewGameRequest = NewGameRequest()) -> GameStateResponse:
    global _game, _player_types, _agents, _recorder
    global _hyp_marker, _hyp_player_types, _hyp_agents
    global _manual_factories, _in_factory_setup, _factory_cursor
    global _last_game_id
    global _search_tree

    _player_types = request.player_types
    _agents = [_make_agent(t) for t in _player_types]
    _search_tree = None
    az_agents = [a for a in _agents if isinstance(a, AlphaZeroAgent)]
    if az_agents:
        _search_tree = SearchTree(
            policy_value_fn=make_policy_value_fn(az_agents[0].net),
            simulations=az_agents[0].simulations,
            temperature=az_agents[0].temperature,
        )
        _search_tree.reset(_game)

    _game = Game()
    _recorder = GameRecorder(
        player_names=list(_player_types), player_types=list(_player_types)
    )
    _history.clear()
    _hyp_marker = None
    _hyp_player_types = None
    _hyp_agents = None
    _manual_factories = request.manual_factories
    _last_game_id = None

    if _manual_factories:
        _enter_factory_setup()
    else:
        _game.setup_round()
        if _recorder is not None:
            _recorder.start_round(_game)
        _in_factory_setup = False
        _factory_cursor = 0

    return _build_response(_game)


@app.post("/agent-move", response_model=GameStateResponse)
def agent_move() -> GameStateResponse:
    global _recorder

    current = _game.current_player_index
    agent = _agents[current]
    if agent is None:
        raise HTTPException(
            status_code=422,
            detail=f"Player {current + 1} is human -- use /move instead",
        )
    move = (
        agent.choose_move(_game, tree=_search_tree)
        if isinstance(agent, AlphaZeroAgent)
        else agent.choose_move(_game)
    )

    if _recorder is not None:
        _recorder.record_move(move, player_index=current)

    _push_history()
    _game.make_move(move)
    if _search_tree is not None:
        _search_tree.advance(move)

    if _hyp_marker is not None:
        _game.advance()
    else:
        _handle_round_end()
        if _game.is_game_over() and _recorder is not None:
            _save_recording(_recorder, _game)
            _recorder = None

    return _build_response(_game)


@app.post("/undo", response_model=GameStateResponse)
def undo() -> GameStateResponse:
    if all(t != "human" for t in _player_types):
        raise HTTPException(
            status_code=400, detail="Undo is not available in bot-vs-bot games"
        )
    floor = _hyp_marker if _hyp_marker is not None else 0
    if len(_history) <= floor:
        raise HTTPException(status_code=400, detail="Nothing to undo")

    _game.__dict__.update(_history.pop().__dict__)
    while len(_history) > floor:
        current = _game.current_player_index
        if _player_types[current] == "human":
            break
        _game.__dict__.update(_history.pop().__dict__)

    return _build_response(_game)


@app.post("/hypothetical/enter", response_model=GameStateResponse)
def hypothetical_enter() -> GameStateResponse:
    global _hyp_marker, _hyp_player_types, _hyp_agents, _player_types, _agents

    if _hyp_marker is not None:
        raise HTTPException(status_code=400, detail="Already in hypothetical mode")

    _hyp_marker = len(_history)
    _hyp_player_types = list(_player_types)
    _hyp_agents = list(_agents)
    _player_types = ["human", "human"]
    _agents = [None, None]

    return _build_response(_game)


@app.post("/hypothetical/commit", response_model=GameStateResponse)
def hypothetical_commit() -> GameStateResponse:
    if _hyp_marker is None:
        raise HTTPException(status_code=400, detail="Not in hypothetical mode")
    _exit_hypothetical(keep_state=True)
    return _build_response(_game)


@app.post("/hypothetical/discard", response_model=GameStateResponse)
def hypothetical_discard() -> GameStateResponse:
    if _hyp_marker is None:
        raise HTTPException(status_code=400, detail="Not in hypothetical mode")
    _exit_hypothetical(keep_state=False)
    return _build_response(_game)


@app.post("/hypothetical/from-snapshot", response_model=GameStateResponse)
def hypothetical_from_snapshot(
    request: HypotheticalSnapshotRequest,
) -> GameStateResponse:
    global _hyp_marker, _hyp_player_types, _hyp_agents, _player_types, _agents

    if _hyp_marker is not None:
        raise HTTPException(status_code=400, detail="Already in hypothetical mode")

    _push_history()
    _hyp_marker = len(_history) - 1
    _hyp_player_types = list(_player_types)
    _hyp_agents = list(_agents)
    _player_types = ["human", "human"]
    _agents = [None, None]

    scratch = _game_from_snapshot(request)

    for factory, scratch_factory in zip(_game.factories, scratch.factories):
        factory.clear()
        factory.extend(scratch_factory)

    _game.center.clear()
    _game.center.extend(scratch.center)
    _game.current_player_index = scratch.current_player_index

    for player, scratch_player in zip(_game.players, scratch.players):
        player.score = scratch_player.score
        player.pending = scratch_player.pending
        player.penalty = scratch_player.penalty
        player.bonus = scratch_player.bonus
        player.pattern_grid = [row[:] for row in scratch_player.pattern_grid]
        player.wall = [row[:] for row in scratch_player.wall]
        player.floor_line = scratch_player.floor_line[:]

    return _build_response(_game)


@app.post("/hypothetical/replace-snapshot", response_model=GameStateResponse)
def hypothetical_replace_snapshot(
    request: HypotheticalSnapshotRequest,
) -> GameStateResponse:
    if _hyp_marker is None:
        raise HTTPException(status_code=400, detail="Not in hypothetical mode")

    _game.current_player_index = request.current_player

    for factory, tile_names in zip(_game.factories, request.factories):
        factory.clear()
        for name in tile_names:
            factory.append(_str_to_tile(name))

    _game.center.clear()
    for name in request.center:
        _game.center.append(_str_to_tile(name))

    for player, board_req in zip(_game.players, request.boards):
        player.score = board_req.score
        _decode_pattern_lines(player, board_req.pattern_lines)
        player.wall = _decode_wall(board_req.wall)
        player.floor_line = [_str_to_tile(name) for name in board_req.floor_line]
        player._update_pending()
        player._update_penalty()
        player._update_bonus()

    return _build_response(_game)


@app.post("/setup-factories/start", response_model=GameStateResponse)
def setup_factories_start() -> GameStateResponse:
    _enter_factory_setup()
    return _build_response(_game)


@app.post("/setup-factories/place", response_model=GameStateResponse)
def setup_factories_place(request: PlaceTileRequest) -> GameStateResponse:
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    if _factory_cursor >= _total_slots():
        raise HTTPException(status_code=400, detail="All factory slots are full")

    factory_index, _ = _flat_to_factory_slot(_factory_cursor)
    if len(_game.factories[factory_index]) >= 4:
        raise HTTPException(status_code=400, detail="Target factory is already full")

    tile = _str_to_setup_tile(request.color)
    _draw_one(tile)

    _game.factories[factory_index].append(tile)
    _factory_cursor += 1

    return _build_response(_game)


@app.post("/setup-factories/remove", response_model=GameStateResponse)
def setup_factories_remove(request: RemoveTileRequest) -> GameStateResponse:
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    factory = _game.factories[request.factory]
    if request.slot >= len(factory):
        raise HTTPException(status_code=400, detail="No tile in that slot")

    tile = factory.pop(request.slot)
    _game.bag.append(tile)

    _factory_cursor = request.factory * 4 + request.slot

    return _build_response(_game)


@app.post("/setup-factories/restart", response_model=GameStateResponse)
def setup_factories_restart() -> GameStateResponse:
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    for factory in _game.factories:
        _game.bag.extend(factory)
        factory.clear()

    random.shuffle(_game.bag)
    _factory_cursor = 0
    return _build_response(_game)


@app.post("/setup-factories/random", response_model=GameStateResponse)
def setup_factories_random() -> GameStateResponse:
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    if sum(len(f) for f in _game.factories) == _total_slots():
        for factory in _game.factories:
            _game.bag.extend(factory)
            factory.clear()
        random.shuffle(_game.bag)
        _factory_cursor = 0

    for factory in _game.factories:
        while len(factory) < 4:
            if not _game.bag:
                if not _game.discard:
                    logger.debug("no tiles remaining to fill factories")
                    break
                _game.bag.extend(_game.discard)
                _game.discard.clear()
                random.shuffle(_game.bag)
            factory.append(_game.bag.pop())

    _factory_cursor = _total_slots()
    return _build_response(_game)


@app.post("/setup-factories/commit", response_model=GameStateResponse)
def setup_factories_commit() -> GameStateResponse:
    global _in_factory_setup, _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    total_filled = sum(len(f) for f in _game.factories)
    if total_filled < _total_slots():
        raise HTTPException(
            status_code=400,
            detail=f"Factories not full: {total_filled}/{_total_slots()} slots filled",
        )

    _in_factory_setup = False
    _factory_cursor = 0
    if _recorder is not None:
        _recorder.start_round(_game)
    return _build_response(_game)


@app.post("/setup-factories/load", response_model=GameStateResponse)
def setup_factories_load(request: list[list[str]]) -> GameStateResponse:
    global _in_factory_setup, _factory_cursor

    if len(request) != len(_game.factories):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(_game.factories)} factories, got {len(request)}",
        )

    total = sum(len(f) for f in request)
    if total != _total_slots():
        raise HTTPException(
            status_code=400,
            detail=f"Expected {_total_slots()} tiles total, got {total}",
        )

    for factory in _game.factories:
        _game.bag.extend(factory)
        factory.clear()
    random.shuffle(_game.bag)

    for factory_req, factory in zip(request, _game.factories):
        for name in factory_req:
            tile = _str_to_setup_tile(name)
            _draw_one(tile)
            factory.append(tile)

    _in_factory_setup = False
    _factory_cursor = 0
    if _recorder is not None:
        _recorder.start_round(_game)

    return _build_response(_game)


@app.get("/recordings", response_model=list[RecordingSummary])
def list_recordings() -> list[RecordingSummary]:
    folders = {
        "human": _RECORDINGS_DIR,
        "eval": _RECORDINGS_DIR / "eval",
        "training": _RECORDINGS_DIR / "training",
    }
    summaries = []
    for folder_name, folder_path in folders.items():
        if not folder_path.exists():
            continue
        for path in folder_path.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(
                    RecordingSummary(
                        game_id=data["game_id"],
                        timestamp=data["timestamp"],
                        player_names=data["player_names"],
                        final_scores=data.get("final_scores", []),
                        winner=data.get("winner"),
                        folder=folder_name,
                    )
                )
            except Exception:
                logger.warning("skipping unreadable recording: %s", path.name)
    summaries.sort(key=lambda s: s.timestamp, reverse=True)
    return summaries


@app.get("/recordings/{game_id}")
def get_recording(game_id: str) -> dict:
    folders = [
        _RECORDINGS_DIR,
        _RECORDINGS_DIR / "eval",
        _RECORDINGS_DIR / "training",
    ]
    for folder_path in folders:
        if not folder_path.exists():
            continue
        for path in folder_path.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("game_id") == game_id:
                    record = GameRecord.from_dict(data)
                    computed_turns, final_boards = record.reconstruct()
                    return {
                        **data,
                        "computed_turns": computed_turns,
                        "final_boards": final_boards,
                    }
            except Exception:
                logger.exception("failed to load or reconstruct recording %s", game_id)
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to reconstruct recording {game_id!r}",
                )
    raise HTTPException(status_code=404, detail=f"Recording {game_id!r} not found")


# endregion

# region Inspector helpers -------------------------------------------------

_ALPHABETA_PRESETS: dict[str, dict] = {
    "alphabeta_easy": {"depth": 1, "threshold": 4},
    "alphabeta_medium": {"depth": 2, "threshold": 6},
    "alphabeta_hard": {"depth": 3, "threshold": 8},
    "alphabeta_extreme": {"depth": 4, "threshold": 10},
}


def _make_alphabeta_pv(agent_name: str) -> PolicyValueFn:
    config = _ALPHABETA_PRESETS[agent_name]
    ab_agent = AlphaBetaAgent(
        depth=config["depth"],
        threshold=config["threshold"],
    )

    def alphabeta_pv(game: Game, legal: list[Move]) -> tuple[list[float], float]:
        if not legal:
            return [], 0.0
        ab_agent.choose_move(game)
        distribution = ab_agent.policy_distribution(game)
        dist_moves = [move for move, _ in distribution]
        dist_priors = [prior for _, prior in distribution]
        priors = []
        for legal_move in legal:
            matched = next(
                (
                    dist_priors[i]
                    for i, dist_move in enumerate(dist_moves)
                    if dist_move == legal_move
                ),
                0.0,
            )
            priors.append(matched)
        return priors, 0.0

    return alphabeta_pv


def _make_minimax_pv() -> PolicyValueFn:
    def minimax_pv(game: Game, legal: list[Move]) -> tuple[list[float], float]:
        if not legal:
            return [], 0.0
        n = len(legal)
        priors = [1.0 / n] * n
        current_index = game.current_player_index
        my_earned = game.players[current_index].earned
        opp_earned = game.players[1 - current_index].earned
        value = (my_earned - opp_earned) / 50.0
        return priors, value

    return minimax_pv


def _make_alphazero_pv(simulations: int) -> PolicyValueFn | None:
    checkpoint_path = Path("checkpoints/latest.pt")
    if not checkpoint_path.exists():
        logger.warning(
            "alphazero inspector requested but no checkpoint found at %s",
            checkpoint_path,
        )
        return None

    import torch
    from neural.model import AzulNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    net = AzulNet()
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return make_policy_value_fn(net)


def _build_inspector_tree(agent_name: str, simulations: int) -> SearchTree:
    if agent_name == "minimax":
        policy_value_fn = _make_minimax_pv()
        use_heuristic_value = True
        effective_simulations = _INSPECTOR_BATCH

    elif agent_name in _ALPHABETA_PRESETS:
        policy_value_fn = _make_alphabeta_pv(agent_name)
        use_heuristic_value = True
        effective_simulations = _INSPECTOR_BATCH

    elif agent_name == "alphazero":
        az_pv = _make_alphazero_pv(simulations)
        if az_pv is None:
            policy_value_fn = _uniform_pv
            use_heuristic_value = True
        else:
            policy_value_fn = az_pv
            use_heuristic_value = False
        effective_simulations = _INSPECTOR_BATCH

    else:
        logger.warning(
            "unknown inspector agent %r, falling back to uniform", agent_name
        )
        policy_value_fn = _uniform_pv
        use_heuristic_value = True
        effective_simulations = _INSPECTOR_BATCH

    return SearchTree(
        policy_value_fn=policy_value_fn,
        simulations=effective_simulations,
        use_heuristic_value=use_heuristic_value,
    )


def _inspector_snapshot() -> dict:
    assert _inspector_tree is not None
    root = _inspector_tree._root
    perspective = f"P{root.game.current_player_index + 1}" if root is not None else "P1"
    sim_count = root.visits if root is not None else 0
    return {
        "game_id": _inspector_game_id,
        "move_index": _inspector_move_index,
        "sim_count": sim_count,
        "done": _inspector_tree.is_stable(),
        "tree": _inspector_tree.serialize(),
        "checkpoint": _inspector_snapshot_agent,
        "perspective": perspective,
    }


def _inspector_init(
    game: Game,
    game_id: str,
    move_index: int,
    agent_name: str = "minimax",
    simulations: int = 200,
) -> None:
    global _inspector_tree, _inspector_game_id
    global _inspector_move_index, _inspector_sim_count
    global _inspector_snapshot_agent, _inspector_snapshot_sims

    _inspector_tree = _build_inspector_tree(agent_name, simulations)
    _inspector_tree.reset(game)
    _inspector_game_id = game_id
    _inspector_move_index = move_index
    _inspector_sim_count = 0
    _inspector_snapshot_agent = agent_name
    _inspector_snapshot_sims = simulations


def _inspector_load(
    game_id: str,
    move_index: int,
    agent_name: str = "minimax",
    simulations: int = 200,
) -> None:
    folders = [
        _RECORDINGS_DIR,
        _RECORDINGS_DIR / "eval",
        _RECORDINGS_DIR / "training",
    ]
    record = None
    for folder in folders:
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("game_id") == game_id:
                    record = GameRecord.from_dict(data)
                    break
            except Exception:
                continue
        if record is not None:
            break

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Recording {game_id!r} not found",
        )

    total_moves = sum(len(r.moves) for r in record.rounds)
    if move_index < 0 or move_index > total_moves:
        raise HTTPException(
            status_code=422,
            detail=f"move_index {move_index} out of range [0, {total_moves}]",
        )

    game = replay_to_move(record, move_index)
    _inspector_init(
        game, game_id, move_index, agent_name=agent_name, simulations=simulations
    )


def _inspector_run_batch() -> None:
    assert _inspector_tree is not None
    if _inspector_tree.is_stable():
        return
    _inspector_tree._run_simulations()
    _inspector_tree.record_batch_stability()


# endregion

# region Inspector endpoints -----------------------------------------------


@app.get("/inspect/{game_id}/{move_index}/state")
def inspect_state(
    game_id: str,
    move_index: int,
    agent: str = "minimax",
    simulations: int = 200,
) -> dict:
    position_changed = (
        _inspector_tree is None
        or _inspector_game_id != game_id
        or _inspector_move_index != move_index
    )
    config_changed = (
        _inspector_snapshot_agent != agent or _inspector_snapshot_sims != simulations
    )
    if position_changed or config_changed:
        _inspector_load(game_id, move_index, agent_name=agent, simulations=simulations)
        _inspector_run_batch()
    return _inspector_snapshot()


@app.post("/inspect/extend")
def inspect_extend() -> dict:
    if _inspector_tree is None:
        raise HTTPException(
            status_code=404,
            detail="No active inspector tree — call "
            "/inspect/{game_id}/{move_index}/state first",
        )
    _inspector_run_batch()
    return _inspector_snapshot()


@app.post("/inspect/reset")
def inspect_reset() -> dict:
    global _inspector_tree, _inspector_game_id
    global _inspector_move_index, _inspector_sim_count
    global _inspector_snapshot_agent, _inspector_snapshot_sims

    _inspector_tree = None
    _inspector_game_id = None
    _inspector_move_index = None
    _inspector_sim_count = 0
    _inspector_snapshot_agent = "minimax"
    _inspector_snapshot_sims = 200
    return {"cleared": True}


@app.post("/inspect/live")
def inspect_live(
    request: HypotheticalSnapshotRequest,
    agent: str = "minimax",
    simulations: int = 200,
) -> dict:
    game = _game_from_snapshot(request)
    _inspector_init(game, "live", 0, agent_name=agent, simulations=simulations)
    _inspector_run_batch()
    return _inspector_snapshot()


@app.get("/inspect/live/state")
def inspect_live_state() -> dict:
    if _inspector_tree is None or _inspector_game_id != "live":
        raise HTTPException(
            status_code=404,
            detail="No active live inspector tree",
        )
    return _inspector_snapshot()


# endregion
