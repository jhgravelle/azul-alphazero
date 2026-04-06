# engine/game_recorder.py

"""Game recorder for capturing Azul games for replay and analysis."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.board import Board
from engine.game import Game, Move
from engine.constants import PLAYERS


# ── Board state capture ────────────────────────────────────────────────────


def _capture_board(board: Board) -> dict[str, Any]:
    """Return a human-readable snapshot of a player board.

    Tile enum values are converted to their string names (e.g. 'BLUE').
    Empty wall cells are stored as None.
    """
    return {
        "score": board.score,
        "pattern_lines": [[tile.name for tile in line] for line in board.pattern_lines],
        "wall": [
            [cell.name if cell is not None else None for cell in row]
            for row in board.wall
        ],
        "floor_line": [tile.name for tile in board.floor_line],
    }


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class TurnRecord:
    """A snapshot of one turn: the board state before the move, and the move itself.

    Attributes:
        player_index: Which player took this turn.
        board_states: One snapshot dict per player, captured before the move.
        move_source: Factory index, or CENTER (-1), or FLOOR (-2).
        move_tile: Tile color name (e.g. 'BLUE').
        move_destination: Pattern line index, or FLOOR (-2).
        analysis: Optional agent-specific data (MCTS visits, value estimates, etc.).
    """

    player_index: int
    board_states: list[dict[str, Any]]
    move_source: int
    move_tile: str
    move_destination: int
    analysis: dict[str, Any] | None = None


@dataclass
class GameRecord:
    """The complete record of one Azul game.

    Attributes:
        game_id: Unique identifier for this game.
        timestamp: ISO 8601 UTC timestamp of when recording began.
        player_names: Display name for each player.
        turns: Ordered list of turn records.
        final_scores: Score for each player after end-of-game bonuses.
        winner: Index of the winning player, or None if not yet finalized.
    """

    game_id: str
    timestamp: str
    player_names: list[str]
    turns: list[TurnRecord] = field(default_factory=list)
    final_scores: list[int] = field(default_factory=list)
    winner: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a JSON-compatible dict."""
        return {
            "game_id": self.game_id,
            "timestamp": self.timestamp,
            "player_names": self.player_names,
            "turns": [
                {
                    "player_index": turn.player_index,
                    "board_states": turn.board_states,
                    "move_source": turn.move_source,
                    "move_tile": turn.move_tile,
                    "move_destination": turn.move_destination,
                    "analysis": turn.analysis,
                }
                for turn in self.turns
            ],
            "final_scores": self.final_scores,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameRecord":
        """Deserialize a GameRecord from a dict (e.g. loaded from JSON)."""
        turns = [
            TurnRecord(
                player_index=turn["player_index"],
                board_states=turn["board_states"],
                move_source=turn["move_source"],
                move_tile=turn["move_tile"],
                move_destination=turn["move_destination"],
                analysis=turn.get("analysis"),
            )
            for turn in data.get("turns", [])
        ]
        return cls(
            game_id=data["game_id"],
            timestamp=data["timestamp"],
            player_names=data["player_names"],
            turns=turns,
            final_scores=data.get("final_scores", []),
            winner=data.get("winner"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GameRecord":
        """Load a GameRecord from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ── Recorder ───────────────────────────────────────────────────────────────


class GameRecorder:
    """Records an Azul game turn by turn for later replay and analysis.

    Usage::

        recorder = GameRecorder(player_names=["Alice", "Bob"])
        # Before each move:
        recorder.record_turn(game, move, analysis={"value_estimate": 0.6})
        game.make_move(move)
        # After game ends:
        recorder.finalize(game)
        recorder.save("recordings/game_001.json")

    Args:
        player_names: Display name for each player. Defaults to
            ["Player 0", "Player 1"].
    """

    def __init__(self, player_names: list[str] | None = None) -> None:
        if player_names is None:
            player_names = [f"Player {i}" for i in range(PLAYERS)]
        self.player_names = player_names
        self.record = GameRecord(
            game_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            player_names=player_names,
        )

    def record_turn(
        self,
        game: Game,
        move: Move,
        analysis: dict[str, Any] | None = None,
    ) -> None:
        """Capture the board state and the chosen move.

        Must be called BEFORE game.make_move(move) so the snapshot reflects
        the state the decision was made from.

        Args:
            game: The current game instance.
            move: The move about to be played.
            analysis: Optional agent-specific data to attach to this turn.
        """
        board_states = [_capture_board(player) for player in game.state.players]
        turn = TurnRecord(
            player_index=game.state.current_player,
            board_states=board_states,
            move_source=move.source,
            move_tile=move.tile.name,
            move_destination=move.destination,
            analysis=analysis,
        )
        self.record.turns.append(turn)

    def finalize(self, game: Game) -> None:
        """Record final scores and determine the winner.

        Call this after game.score_game() has been called.

        Args:
            game: The completed game instance.
        """
        self.record.final_scores = [p.score for p in game.state.players]
        self.record.winner = max(
            range(len(self.record.final_scores)),
            key=lambda i: self.record.final_scores[i],
        )

    def to_json(self) -> str:
        """Serialize the full game record to a JSON string."""
        return json.dumps(self.record.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        """Write the game record to a JSON file.

        Args:
            path: Destination file path. Parent directories must exist.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
