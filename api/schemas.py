# api/schemas.py

"""Pydantic schemas for the Azul API request and response bodies."""

from pydantic import BaseModel


class MoveRequest(BaseModel):
    """A move — used both for submitting moves and describing legal moves."""

    source: int  # factory index, or -1 for the center pool
    color: str  # tile color as a string e.g. "BLUE"
    destination: int  # pattern line row (0–4), or -2 for the floor


class BoardResponse(BaseModel):
    """The state of one player's board."""

    score: int
    pattern_lines: list[list[str]]  # e.g. [["BLUE"], [], ["RED", "RED"], ...]
    wall: list[list[str | None]]  # None = empty cell
    floor_line: list[str]


class GameStateResponse(BaseModel):
    """The complete game state sent to the frontend after every action."""

    current_player: int
    factories: list[list[str]]
    center: list[str]
    boards: list[BoardResponse]
    is_game_over: bool
    winner: int | None
    legal_moves: list[MoveRequest]
