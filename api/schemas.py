# api/schemas.py

"""Pydantic schemas for the Azul API request and response bodies."""

from typing import Literal
from pydantic import BaseModel, field_validator

PlayerType = Literal["human", "random"]


class MoveRequest(BaseModel):
    """A move — used both for submitting moves and describing legal moves."""

    source: int
    color: str
    destination: int


class NewGameRequest(BaseModel):
    """Configuration for starting a new game."""

    player_types: list[PlayerType] = ["human", "human"]

    @field_validator("player_types")
    @classmethod
    def validate_player_types(cls, v: list[str]) -> list[str]:
        if len(v) != 2:
            raise ValueError("player_types must have exactly 2 entries")
        return v


class BoardResponse(BaseModel):
    """The state of one player's board."""

    score: int
    pattern_lines: list[list[str]]
    wall: list[list[str | None]]
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
    player_types: list[PlayerType]
