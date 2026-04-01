# api/main.py

"""FastAPI application for the Azul game."""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from engine.game import Game, Move
from engine.tile import Tile
from api.schemas import BoardResponse, GameStateResponse, MoveRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="Azul API")

# Allow the HTML frontend (opened as a local file) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global game instance — simple and sufficient for Phase 2.
_game = Game()
_game.setup_round()


def _tile_to_str(tile: Tile) -> str:
    """Convert a Tile enum value to its string name."""
    return tile.name


def _str_to_tile(name: str) -> Tile:
    """Convert a color string to a Tile enum value."""
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
        winner = max(
            range(len(game.state.players)), key=lambda i: game.state.players[i].score
        )

    legal = [
        MoveRequest(
            source=m.source,
            color=m.color.name,
            destination=m.destination,
        )
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
        source=move_request.source,
        color=color,
        destination=move_request.destination,
    )
    legal = _game.legal_moves()
    if move not in legal:
        raise HTTPException(status_code=422, detail="Illegal move")
    _game.make_move(move)
    return _build_response(_game)


@app.post("/new-game", response_model=GameStateResponse)
def new_game() -> GameStateResponse:
    """Reset the game and return the fresh state."""
    global _game
    _game = Game()
    _game.setup_round()
    return _build_response(_game)
