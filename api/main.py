# api/main.py

"""FastAPI application for the Azul game."""

import copy
import json
import logging
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
    MoveRequest,
    NewGameRequest,
    PendingBonus,
    PendingPlacement,
    PlayerType,
    RecordingSummary,
)
from engine.board import Board
from engine.constants import Tile
from engine.game import Game, Move
from engine.game_recorder import GameRecorder
from engine.game_state import GameState
from engine.scoring import (
    pending_bonus_details,
    pending_placement_details,
)

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
# _hyp_marker: index into _history where hypothetical mode began.
# _hyp_player_types / _hyp_agents: the real config to restore on commit or discard.
_hyp_marker: int | None = None
_hyp_player_types: list[PlayerType] | None = None
_hyp_agents: list[Agent | None] | None = None


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
        case _:
            return None


def _tile_to_str(tile: Tile) -> str:
    return tile.name


def _str_to_tile(name: str) -> Tile:
    try:
        return Tile[name]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown tile color: {name!r}")


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
    )


def _save_recording(recorder: GameRecorder, game: Game) -> None:
    """Finalize and save the recording to disk. Silently skips on error."""
    try:
        recorder.finalize(game)
        _RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = _RECORDINGS_DIR / f"{recorder.record.game_id}.json"
        recorder.save(path)
        logger.info("saved recording %s", recorder.record.game_id)
    except Exception:
        logger.exception("failed to save recording")


def _exit_hypothetical(*, keep_state: bool) -> None:
    """Shared teardown for commit and discard.

    keep_state=True  -> commit: history above marker is dropped, state stays.
    keep_state=False -> discard: history is truncated to marker, state restored.
    """
    global _hyp_marker, _hyp_player_types, _hyp_agents, _player_types, _agents

    marker = _hyp_marker
    saved_types = _hyp_player_types
    saved_agents = _hyp_agents

    assert marker is not None

    # Clear hypothetical state first.
    _hyp_marker = None
    _hyp_player_types = None
    _hyp_agents = None

    # These are always set together with _hyp_marker, so they cannot be None here.
    assert saved_types is not None
    assert saved_agents is not None

    # Restore real player config.
    _player_types = saved_types
    _agents = saved_agents

    if keep_state:
        # Commit: drop history snapshots taken during hypothetical mode.
        del _history[marker:]
    else:
        # Discard: restore game to the snapshot at the marker boundary.
        if marker < len(_history):
            _game.state = _history[marker]
        del _history[marker:]


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
        _recorder.record_turn(_game, move)

    _push_history()
    _game.make_move(move)

    if _game.is_game_over() and _recorder is not None:
        _save_recording(_recorder, _game)
        _recorder = None

    return _build_response(_game)


@app.post("/new-game", response_model=GameStateResponse)
def new_game(request: NewGameRequest = NewGameRequest()) -> GameStateResponse:
    """Reset the game with the given player configuration."""
    global _game, _player_types, _agents, _recorder
    global _hyp_marker, _hyp_player_types, _hyp_agents

    _player_types = request.player_types
    _agents = [_make_agent(t) for t in _player_types]
    _game = Game()
    _game.setup_round()
    _recorder = GameRecorder(player_names=list(_player_types))
    _history.clear()
    _hyp_marker = None
    _hyp_player_types = None
    _hyp_agents = None
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
    move = agent.choose_move(_game)

    if _recorder is not None:
        _recorder.record_turn(_game, move)

    _push_history()
    _game.make_move(move)

    if _game.is_game_over() and _recorder is not None:
        _save_recording(_recorder, _game)
        _recorder = None

    return _build_response(_game)


@app.post("/undo", response_model=GameStateResponse)
def undo() -> GameStateResponse:
    """Restore the game to the state before the last move."""
    if all(t != "human" for t in _player_types):
        raise HTTPException(
            status_code=400, detail="Undo is not available in bot-vs-bot games"
        )
    # In hypothetical mode, undo cannot pop past the marker.
    floor = _hyp_marker if _hyp_marker is not None else 0
    if len(_history) <= floor:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    _game.state = _history.pop()
    return _build_response(_game)


@app.post("/hypothetical/enter", response_model=GameStateResponse)
def hypothetical_enter() -> GameStateResponse:
    """Enter hypothetical mode -- both players become human, marker is set."""
    global _hyp_marker, _hyp_player_types, _hyp_agents, _player_types, _agents

    if _hyp_marker is not None:
        raise HTTPException(status_code=400, detail="Already in hypothetical mode")

    # Save real config and record where on the stack we are now.
    _hyp_marker = len(_history)
    _hyp_player_types = list(_player_types)
    _hyp_agents = list(_agents)

    # Override both players to human.
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


@app.get("/recordings", response_model=list[RecordingSummary])
def list_recordings() -> list[RecordingSummary]:
    """Return a summary of every saved game, newest first."""
    if not _RECORDINGS_DIR.exists():
        return []
    summaries = []
    for path in sorted(
        _RECORDINGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                RecordingSummary(
                    game_id=data["game_id"],
                    timestamp=data["timestamp"],
                    player_names=data["player_names"],
                    final_scores=data.get("final_scores", []),
                    winner=data.get("winner"),
                )
            )
        except Exception:
            logger.warning("skipping unreadable recording: %s", path.name)
    return summaries


@app.get("/recordings/{game_id}")
def get_recording(game_id: str) -> dict:
    """Return the full JSON record for one game."""
    path = _RECORDINGS_DIR / f"{game_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Recording {game_id!r} not found")
    return json.loads(path.read_text(encoding="utf-8"))
