# api/schemas.py

"""Pydantic schemas for the Azul API request and response bodies."""

from typing import Literal

from pydantic import BaseModel, field_validator

PlayerType = Literal["human", "random", "cautious", "efficient", "greedy", "mcts"]


class MoveRequest(BaseModel):
    """A move -- used both for submitting moves and describing legal moves."""

    source: int
    tile: str
    destination: int


class NewGameRequest(BaseModel):
    """Configuration for starting a new game."""

    player_types: list[PlayerType] = ["human", "greedy"]

    @field_validator("player_types")
    @classmethod
    def must_have_two_players(cls, v: list[str]) -> list[str]:
        if len(v) != 2:
            raise ValueError("player_types must have exactly 2 entries")
        return v


class PendingPlacement(BaseModel):
    """Wall annotation for a single full pattern line awaiting end-of-round scoring."""

    row: int
    column: int
    placement_points: int


class PendingBonus(BaseModel):
    """An end-of-game bonus already guaranteed by the current wall state."""

    bonus_type: Literal["row", "column", "tile"]
    index: int
    bonus_points: int


class BoardResponse(BaseModel):
    """The state of one player's board."""

    score: int
    pattern_lines: list[list[str]]
    wall: list[list[str | None]]
    floor_line: list[str]
    pending_placements: list[PendingPlacement] = []
    pending_bonuses: list[PendingBonus] = []


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
    round: int


# ── Recording schemas ──────────────────────────────────────────────────────


class RecordingSummary(BaseModel):
    """Minimal info about a saved game -- used to populate the replay dropdown."""

    game_id: str
    timestamp: str
    player_names: list[str]
    final_scores: list[int]
    winner: int | None
