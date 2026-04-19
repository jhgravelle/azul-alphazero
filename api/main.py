# api/main.py

"""FastAPI application for the Azul game."""

import copy
from datetime import datetime
import json
import logging
import random
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.base import Agent
from agents.cautious import CautiousAgent
from agents.efficient import EfficientAgent
from agents.greedy import GreedyAgent
from agents.mcts import MCTSAgent
from agents.random import RandomAgent
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
from engine.board import Board
from engine.constants import Tile
from engine.game import Game, Move
from engine.game_recorder import GameRecorder, GameRecord
from engine.game_state import GameState
from engine.replay import replay_to_move
from engine.scoring import (
    pending_bonus_details,
    pending_placement_details,
)
from neural.search_tree import SearchTree, make_policy_value_fn
from agents.alphazero import AlphaZeroAgent

logger = logging.getLogger(__name__)

app = FastAPI(title="Azul API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recordings directory -- one JSON file per completed game.
_RECORDINGS_DIR = Path("recordings")

# Game state -- one game at a time.
_game = Game()
_game.setup_round()
_player_types: list[PlayerType] = ["human", "human"]
_agents: list[Agent | None] = [None, None]
_recorder: GameRecorder | None = None
_history: list[GameState] = []

# Hypothetical mode state.
_hyp_marker: int | None = None
_hyp_player_types: list[PlayerType] | None = None
_hyp_agents: list[Agent | None] | None = None

# Factory setup state.
# _manual_factories: whether this game uses manual factory setup each round.
# _in_factory_setup: whether we are currently in the setup phase.
# _factory_cursor: flat index (0..num_factories*4-1) of the next slot to fill.
_manual_factories: bool = False
_in_factory_setup: bool = False
_factory_cursor: int = 0
_last_game_id: str | None = None
_search_tree: SearchTree | None = None

# Inspector state — one active tree at a time.
_inspector_tree: SearchTree | None = None
_inspector_game_id: str | None = None
_inspector_move_index: int | None = None
_inspector_sim_count: int = 0
_INSPECTOR_BATCH = 200


def _make_agent(player_type: PlayerType) -> Agent | None:
    """Return an Agent instance for the given type, or None for human."""
    match player_type:
        case "random":
            return RandomAgent()
        case "cautious":
            return CautiousAgent()
        case "efficient":
            return EfficientAgent()
        case "greedy":
            return GreedyAgent()
        case "mcts":
            return MCTSAgent()
        # case "alphazero":
        #     return AlphaZeroAgent(...)  # wire in once checkpoint exists
        case _:
            return None


def _uniform_pv(game: Game, legal: list[Move]) -> tuple[list[float], float]:
    """Uniform policy, zero value. Used by the inspector when no checkpoint
    is loaded. MCTS still produces meaningful score-differential values via
    _terminal_value and backpropagation."""
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
    """Parse a tile color for factory setup, raising 400 on unknown names."""
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
    """Push a deep copy of the current game state onto the undo stack."""
    _history.append(copy.deepcopy(_game.state))


def _build_pending(
    board: Board,
) -> tuple[list[PendingPlacement], list[PendingBonus]]:
    """Compute pending placements and bonuses for a board without mutating it."""
    placement_details, temp_wall = pending_placement_details(board)
    bonus_details = pending_bonus_details(temp_wall)

    placements = [
        PendingPlacement(
            row=d.row, column=d.column, placement_points=d.placement_points
        )
        for d in placement_details
    ]
    bonuses = [
        PendingBonus(
            bonus_type=d.bonus_type, index=d.index, bonus_points=d.bonus_points
        )
        for d in bonus_details
    ]
    return placements, bonuses


def _build_response(game: Game) -> GameStateResponse:
    """Translate the engine Game object into a GameStateResponse."""
    boards = []
    for player in game.state.players:
        pattern_lines = [[_tile_to_str(t) for t in row] for row in player.pattern_lines]
        wall = [
            [_tile_to_str(t) if t is not None else None for t in row]
            for row in player.wall
        ]
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
        scores = [p.score for p in game.state.players]
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
        current_player=game.state.current_player,
        factories=[[_tile_to_str(t) for t in f] for f in game.state.factories],
        center=[_tile_to_str(t) for t in game.state.center],
        boards=boards,
        is_game_over=game_over,
        winner=winner,
        legal_moves=legal,
        player_types=_player_types,
        round=game.state.round,
        bag_counts=_counts(game.state.bag),
        discard_counts=_counts(game.state.discard),
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
    """Shared teardown for commit and discard."""
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
            _game.state = _history[marker]
        del _history[marker:]


def _enter_factory_setup() -> None:
    """Clear all factories and enter setup mode. Does not touch the bag."""
    global _in_factory_setup, _factory_cursor
    for factory in _game.state.factories:
        factory.clear()
    _game.state.center = [Tile.FIRST_PLAYER]
    _in_factory_setup = True
    _factory_cursor = 0


def _handle_round_end() -> None:
    """Called after every make_move. If the round just ended, either enter
    factory setup mode or auto-setup the next round."""
    if _game.is_game_over():
        return
    sources_empty = (
        all(len(f) == 0 for f in _game.state.factories) and len(_game.state.center) == 0
    )
    if not sources_empty:
        return
    if _manual_factories:
        _enter_factory_setup()
    else:
        _game.setup_round()
        if _search_tree is not None:
            _search_tree.reset(_game)
        if _recorder is not None:
            _recorder.start_round(_game)


def _total_slots() -> int:
    """Total number of factory slots in the current game."""
    return len(_game.state.factories) * 4


def _draw_one(tile: Tile) -> None:
    """Draw one tile of the given color from the bag, refilling from discard if needed.

    Raises HTTPException 400 if the tile is unavailable even after refill.
    """
    if tile not in _game.state.bag:
        # Try refilling from discard.
        if not _game.state.discard:
            raise HTTPException(
                status_code=400,
                detail=f"No {tile.name} tiles available in bag or discard",
            )
        _game.state.bag.extend(_game.state.discard)
        _game.state.discard.clear()
        random.shuffle(_game.state.bag)
        if tile not in _game.state.bag:
            raise HTTPException(
                status_code=400,
                detail=f"No {tile.name} tiles available in bag or discard",
            )
    _game.state.bag.remove(tile)


def _flat_to_factory_slot(cursor: int) -> tuple[int, int]:
    """Convert a flat cursor index to (factory_index, slot_index)."""
    return cursor // 4, cursor % 4


def _next_cursor() -> int:
    """Return the flat index of the next empty slot, starting from _factory_cursor."""
    for i in range(_factory_cursor, _total_slots()):
        factory_index, slot_index = _flat_to_factory_slot(i)
        if slot_index >= len(_game.state.factories[factory_index]):
            return i
    return _total_slots()


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.get("/state", response_model=GameStateResponse)
def get_state() -> GameStateResponse:
    """Return the current game state."""
    return _build_response(_game)


@app.post("/move", response_model=GameStateResponse)
def make_move(move_request: MoveRequest) -> GameStateResponse:
    """Apply a move and return the updated game state."""
    global _recorder

    tile = _str_to_tile(move_request.tile)
    move = Move(
        source=move_request.source, tile=tile, destination=move_request.destination
    )
    if move not in _game.legal_moves():
        raise HTTPException(status_code=422, detail="Illegal move")

    if _recorder is not None:
        _recorder.record_move(move, player_index=_game.state.current_player)

    _push_history()
    _game.make_move(move)

    if _search_tree is not None:
        _search_tree.advance(move)
    if _hyp_marker is not None:
        pass
    else:
        _handle_round_end()
        if _game.is_game_over() and _recorder is not None:
            _save_recording(_recorder, _game)
            _recorder = None

    return _build_response(_game)


@app.post("/new-game", response_model=GameStateResponse)
def new_game(request: NewGameRequest = NewGameRequest()) -> GameStateResponse:
    """Reset the game with the given player configuration."""
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
    """Ask the current player's agent to pick and apply a move."""
    global _recorder

    current = _game.state.current_player
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
    _handle_round_end()

    if _hyp_marker is not None:
        pass
    else:
        _handle_round_end()
        if _game.is_game_over() and _recorder is not None:
            _save_recording(_recorder, _game)
            _recorder = None

    return _build_response(_game)


@app.post("/undo", response_model=GameStateResponse)
def undo() -> GameStateResponse:
    """Restore the game to the state before the last move made by a human."""
    if all(t != "human" for t in _player_types):
        raise HTTPException(
            status_code=400, detail="Undo is not available in bot-vs-bot games"
        )
    floor = _hyp_marker if _hyp_marker is not None else 0
    if len(_history) <= floor:
        raise HTTPException(status_code=400, detail="Nothing to undo")

    # Pop at least once, then keep popping while the restored state
    # belongs to a bot turn — so the human always lands on their own turn.
    _game.state = _history.pop()
    while len(_history) > floor:
        current = _game.state.current_player
        if _player_types[current] == "human":
            break
        _game.state = _history.pop()

    return _build_response(_game)


@app.post("/hypothetical/enter", response_model=GameStateResponse)
def hypothetical_enter() -> GameStateResponse:
    """Enter hypothetical mode -- both players become human, marker is set."""
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
    """Commit hypothetical moves as the real game state and exit hypothetical mode."""
    if _hyp_marker is None:
        raise HTTPException(status_code=400, detail="Not in hypothetical mode")
    _exit_hypothetical(keep_state=True)
    return _build_response(_game)


@app.post("/hypothetical/discard", response_model=GameStateResponse)
def hypothetical_discard() -> GameStateResponse:
    """Discard all hypothetical moves and restore the state from before entering."""
    if _hyp_marker is None:
        raise HTTPException(status_code=400, detail="Not in hypothetical mode")
    _exit_hypothetical(keep_state=False)
    return _build_response(_game)


@app.post("/hypothetical/from-snapshot", response_model=GameStateResponse)
def hypothetical_from_snapshot(
    request: HypotheticalSnapshotRequest,
) -> GameStateResponse:
    """Load a replay snapshot into the game and enter hypothetical mode.

    The existing game state is pushed onto the history stack so discard
    restores it. Bag and discard are left untouched -- hypotheticals are
    assumed to stay within the current round.
    """
    global _hyp_marker, _hyp_player_types, _hyp_agents, _player_types, _agents

    if _hyp_marker is not None:
        raise HTTPException(status_code=400, detail="Already in hypothetical mode")

    # Push current state so discard can restore it, then set marker to point at it.
    _push_history()
    _hyp_marker = len(_history) - 1
    _hyp_player_types = list(_player_types)
    _hyp_agents = list(_agents)
    _player_types = ["human", "human"]
    _agents = [None, None]

    # Load factories.
    for factory, tile_names in zip(_game.state.factories, request.factories):
        factory.clear()
        for name in tile_names:
            factory.append(_str_to_tile(name))

    # Load center.
    _game.state.center.clear()
    for name in request.center:
        _game.state.center.append(_str_to_tile(name))

    # Load current player.
    _game.state.current_player = request.current_player

    # Load board states.
    for player, board_req in zip(_game.state.players, request.boards):
        player.score = board_req.score
        player.pattern_lines = [
            [_str_to_tile(name) for name in line] for line in board_req.pattern_lines
        ]
        player.wall = [
            [_str_to_tile(name) if name is not None else None for name in row]
            for row in board_req.wall
        ]
        player.floor_line = [_str_to_tile(name) for name in board_req.floor_line]

    return _build_response(_game)


@app.post("/hypothetical/replace-snapshot", response_model=GameStateResponse)
def hypothetical_replace_snapshot(
    request: HypotheticalSnapshotRequest,
) -> GameStateResponse:
    """Replace the current in-hypothetical game state with a new snapshot."""
    if _hyp_marker is None:
        raise HTTPException(status_code=400, detail="Not in hypothetical mode")

    _game.state.current_player = request.current_player

    for factory, tile_names in zip(_game.state.factories, request.factories):
        factory.clear()
        for name in tile_names:
            factory.append(_str_to_tile(name))

    _game.state.center.clear()
    for name in request.center:
        _game.state.center.append(_str_to_tile(name))

    for player, board_req in zip(_game.state.players, request.boards):
        player.score = board_req.score
        player.pattern_lines = [
            [_str_to_tile(name) for name in line] for line in board_req.pattern_lines
        ]
        player.wall = [
            [_str_to_tile(name) if name is not None else None for name in row]
            for row in board_req.wall
        ]
        player.floor_line = [_str_to_tile(name) for name in board_req.floor_line]

    return _build_response(_game)


@app.post("/setup-factories/start", response_model=GameStateResponse)
def setup_factories_start() -> GameStateResponse:
    """Enter factory setup mode -- clear all factories, reset cursor to zero."""
    _enter_factory_setup()
    return _build_response(_game)


@app.post("/setup-factories/place", response_model=GameStateResponse)
def setup_factories_place(request: PlaceTileRequest) -> GameStateResponse:
    """Place one tile of the given color into the next empty factory slot."""
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    if _factory_cursor >= _total_slots():
        raise HTTPException(status_code=400, detail="All factory slots are full")

    factory_index, _ = _flat_to_factory_slot(_factory_cursor)
    if len(_game.state.factories[factory_index]) >= 4:
        raise HTTPException(status_code=400, detail="Target factory is already full")

    tile = _str_to_setup_tile(request.color)
    _draw_one(tile)

    _game.state.factories[factory_index].append(tile)
    _factory_cursor += 1

    return _build_response(_game)


@app.post("/setup-factories/remove", response_model=GameStateResponse)
def setup_factories_remove(request: RemoveTileRequest) -> GameStateResponse:
    """Remove a tile from a factory slot and return it to the bag."""
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    factory = _game.state.factories[request.factory]
    if request.slot >= len(factory):
        raise HTTPException(status_code=400, detail="No tile in that slot")

    tile = factory.pop(request.slot)
    _game.state.bag.append(tile)

    # Move cursor back to the removed slot's flat index.
    _factory_cursor = request.factory * 4 + request.slot

    return _build_response(_game)


@app.post("/setup-factories/restart", response_model=GameStateResponse)
def setup_factories_restart() -> GameStateResponse:
    """Return all placed tiles to the bag and reset the cursor to zero."""
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    for factory in _game.state.factories:
        _game.state.bag.extend(factory)
        factory.clear()

    random.shuffle(_game.state.bag)
    _factory_cursor = 0
    return _build_response(_game)


@app.post("/setup-factories/random", response_model=GameStateResponse)
def setup_factories_random() -> GameStateResponse:
    """Fill all remaining empty factory slots randomly from the bag."""
    global _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    # If factories are already full, restart first to get a fresh shuffle.
    if sum(len(f) for f in _game.state.factories) == _total_slots():
        for factory in _game.state.factories:
            _game.state.bag.extend(factory)
            factory.clear()
        random.shuffle(_game.state.bag)
        _factory_cursor = 0

    for factory in _game.state.factories:
        while len(factory) < 4:
            if not _game.state.bag:
                if not _game.state.discard:
                    logger.debug("no tiles remaining to fill factories")
                    break
                _game.state.bag.extend(_game.state.discard)
                _game.state.discard.clear()
                random.shuffle(_game.state.bag)
            factory.append(_game.state.bag.pop())

    # Stay in setup mode so user can review before committing.
    _factory_cursor = _total_slots()
    return _build_response(_game)


@app.post("/setup-factories/commit", response_model=GameStateResponse)
def setup_factories_commit() -> GameStateResponse:
    """Commit the factory setup and begin the round."""
    global _in_factory_setup, _factory_cursor

    if not _in_factory_setup:
        raise HTTPException(status_code=400, detail="Not in factory setup mode")

    total_filled = sum(len(f) for f in _game.state.factories)
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


@app.get("/recordings", response_model=list[RecordingSummary])
def list_recordings() -> list[RecordingSummary]:
    """Return a summary of every saved game across all subfolders, newest first."""
    folders = {
        "human": _RECORDINGS_DIR,
        "eval": _RECORDINGS_DIR / "eval",
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
    """Return the full record plus pre-computed turn states for one game."""
    folders = {
        "human": _RECORDINGS_DIR,
        "eval": _RECORDINGS_DIR / "eval",
    }
    for folder_path in folders.values():
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


# ── Inspector endpoints ────────────────────────────────────────────────────


def _inspector_snapshot() -> dict:
    """Serialize current inspector state to a JSON-compatible dict."""
    assert _inspector_tree is not None
    return {
        "game_id": _inspector_game_id,
        "move_index": _inspector_move_index,
        "sim_count": _inspector_sim_count,
        "done": _inspector_tree.is_stable(),
        "tree": _inspector_tree.serialize(),
        "checkpoint": "uniform",
    }


def _inspector_load(game_id: str, move_index: int) -> None:
    """Build a fresh inspector tree rooted at the given position."""
    global _inspector_tree, _inspector_game_id
    global _inspector_move_index, _inspector_sim_count

    # Find the recording.
    folders = [_RECORDINGS_DIR, _RECORDINGS_DIR / "eval"]
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

    _inspector_tree = SearchTree(
        policy_value_fn=_uniform_pv,
        simulations=_INSPECTOR_BATCH,
    )
    _inspector_tree.reset(game)
    _inspector_game_id = game_id
    _inspector_move_index = move_index
    _inspector_sim_count = 0


def _inspector_run_batch() -> None:
    global _inspector_sim_count
    assert _inspector_tree is not None
    if _inspector_tree.is_stable():
        return
    _inspector_tree._run_simulations()
    _inspector_tree.record_batch_stability()
    _inspector_sim_count += _INSPECTOR_BATCH


@app.get("/inspect/{game_id}/{move_index}/state")
def inspect_state(game_id: str, move_index: int) -> dict:
    """Return the current inspector tree snapshot.

    If the requested position differs from the active inspector tree,
    a fresh tree is created and one batch of simulations is run before
    returning. If the same position is requested again, the existing tree
    is returned as-is.
    """
    global _inspector_tree

    if (
        _inspector_tree is None
        or _inspector_game_id != game_id
        or _inspector_move_index != move_index
    ):
        _inspector_load(game_id, move_index)
        _inspector_run_batch()

    return _inspector_snapshot()


@app.post("/inspect/extend")
def inspect_extend() -> dict:
    """Run another batch of simulations on the active inspector tree.

    Returns 404 if no inspector tree is active.
    """
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
    """Clear the active inspector tree. Used in testing and UI navigation."""
    global _inspector_tree, _inspector_game_id
    global _inspector_move_index, _inspector_sim_count

    _inspector_tree = None
    _inspector_game_id = None
    _inspector_move_index = None
    _inspector_sim_count = 0
    return {"cleared": True}
