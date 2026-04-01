# api/main.py

"""FastAPI application for the Azul game."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.base import Agent
from agents.random import RandomAgent
from engine.game import Game, Move
from engine.tile import Tile
from api.schemas import (
    BoardResponse,
    GameStateResponse,
    MoveRequest,
    NewGameRequest,
    PlayerType,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Azul API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game state — one game at a time.
_game = Game()
_game.setup_round()
_player_types: list[PlayerType] = ["human", "human"]


def _make_agent(player_type: PlayerType) -> Agent | None:
    """Return an Agent instance for the given type, or None for human."""
    if player_type == "random":
        return RandomAgent()
    return None


_agents: list[Agent | None] = [_make_agent(t) for t in _player_types]


def _tile_to_str(tile: Tile) -> str:
    return tile.name


def _str_to_tile(name: str) -> Tile:
    try:
        return Tile[name]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Unknown tile color: {name!r}")


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
        boards.append(
            BoardResponse(
                score=player.score,
                pattern_lines=pattern_lines,
                wall=wall,
                floor_line=floor_line,
            )
        )

    game_over = game.is_game_over()
    winner = None
    if game_over:
        scores = [p.score for p in game.state.players]
        if scores.count(max(scores)) == 1:
            winner = scores.index(max(scores))
        # else winner stays None — it's a tie

    legal = [
        MoveRequest(source=m.source, color=m.color.name, destination=m.destination)
        for m in game.legal_moves()
    ]

    return GameStateResponse(
        current_player=game.state.current_player,
        factories=[[_tile_to_str(t) for t in f] for f in game.state.factories],
        center=[_tile_to_str(t) for t in game.state.center],
        boards=boards,
        is_game_over=game_over,
        winner=winner,
        legal_moves=legal,
        player_types=_player_types,
        round=game.state.round,  # add this
    )


@app.get("/state", response_model=GameStateResponse)
def get_state() -> GameStateResponse:
    """Return the current game state."""
    return _build_response(_game)


@app.post("/move", response_model=GameStateResponse)
def make_move(move_request: MoveRequest) -> GameStateResponse:
    """Apply a move and return the updated game state."""
    color = _str_to_tile(move_request.color)
    move = Move(
        source=move_request.source, color=color, destination=move_request.destination
    )
    if move not in _game.legal_moves():
        raise HTTPException(status_code=422, detail="Illegal move")
    _game.make_move(move)
    return _build_response(_game)


@app.post("/new-game", response_model=GameStateResponse)
def new_game(request: NewGameRequest = NewGameRequest()) -> GameStateResponse:
    """Reset the game with the given player configuration."""
    global _game, _player_types, _agents
    _player_types = request.player_types
    _agents = [_make_agent(t) for t in _player_types]
    _game = Game()
    _game.setup_round()
    return _build_response(_game)


@app.post("/agent-move", response_model=GameStateResponse)
def agent_move() -> GameStateResponse:
    """Ask the current player's agent to pick and apply a move."""
    current = _game.state.current_player
    agent = _agents[current]
    if agent is None:
        raise HTTPException(
            status_code=422,
            detail=f"Player {current + 1} is human — use /move instead",
        )
    move = agent.choose_move(_game)
    _game.make_move(move)
    return _build_response(_game)
