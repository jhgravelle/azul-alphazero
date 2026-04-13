# engine/game_recorder.py

"""Game recorder for capturing Azul games for replay and analysis."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.board import Board
from engine.constants import PLAYERS, Tile
from engine.game import Game, Move
from engine.scoring import earned_score

# ── Data classes ───────────────────────────────────────────────────────────


@dataclass
class MoveRecord:
    """A single move within a round.

    Attributes:
        player_index: Which player made this move.
        source:       Factory index, CENTER (-1), or FLOOR (-2).
        tile:         Tile color name (e.g. 'BLUE').
        destination:  Pattern line index, or FLOOR (-2).
    """

    player_index: int
    source: int
    tile: str
    destination: int


@dataclass
class RoundRecord:
    """The factory layout and moves for one round.

    Attributes:
        round:     Round number (1-indexed).
        factories: Tile names in each factory at the start of the round.
        center:    Tile names in the center at the start of the round.
        moves:     Ordered list of moves made this round.
    """

    round: int
    factories: list[list[str]]
    center: list[str]
    moves: list[MoveRecord] = field(default_factory=list)


def _board_to_dict(board: Board) -> dict[str, Any]:
    """Serialize a Board to a JSON-compatible dict."""
    return {
        "score": board.score,
        "pattern_lines": [[tile.name for tile in line] for line in board.pattern_lines],
        "wall": [
            [cell.name if cell is not None else None for cell in row]
            for row in board.wall
        ],
        "floor_line": [tile.name for tile in board.floor_line],
    }


def _board_to_dict_with_pending(board: Board) -> dict[str, Any]:
    """Serialize a board including pending placement and bonus details."""
    from engine.scoring import pending_placement_details, pending_bonus_details

    placement_details, temp_wall = pending_placement_details(board)
    bonus_details = pending_bonus_details(temp_wall)
    base = _board_to_dict(board)
    base["pending_placements"] = [
        {"row": d.row, "column": d.column, "placement_points": d.placement_points}
        for d in placement_details
    ]
    base["pending_bonuses"] = [
        {"bonus_type": d.bonus_type, "index": d.index, "bonus_points": d.bonus_points}
        for d in bonus_details
    ]
    return base


def _counts(tile_list: list[Tile]) -> dict[str, int]:
    colors = [t for t in Tile if t != Tile.FIRST_PLAYER]
    return {t.name: tile_list.count(t) for t in colors}


@dataclass
class GameRecord:
    """The complete record of one Azul game.

    Attributes:
        game_id:      Unique identifier for this game.
        timestamp:    ISO 8601 UTC timestamp of when recording began.
        player_names: Display name for each player.
        player_types: Agent type string for each player.
        rounds:       One RoundRecord per round played.
        final_scores: Score for each player after end-of-game bonuses.
        winner:       Index of the winning player, or None for a draw.
    """

    game_id: str
    timestamp: str
    player_names: list[str]
    player_types: list[str] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    final_scores: list[int] = field(default_factory=list)
    winner: int | None = None

    # ── Reconstruction ─────────────────────────────────────────────────

    def reconstruct(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Replay all moves and return (computed_turns, final_boards).

        computed_turns is a flat list — one entry per move across all rounds,
        in order. Each entry reflects the state AFTER the move:

            {
                "round":        int,
                "player_index": int,
                "source":       int,
                "tile":         str,
                "destination":  int,
                "boards":       [board_dict, ...],
                "factories":    [[tile_name, ...], ...],
                "center":       [tile_name, ...],
                "bag_counts":   {color: count, ...},
                "discard_counts": {color: count, ...},
                "grand_totals": [int, ...],
            }

        final_boards is a list of board dicts after end-of-game scoring.
        """
        game = Game()
        computed_turns: list[dict[str, Any]] = []

        for round_record in self.rounds:
            explicit_factories = [
                [Tile[name] for name in factory] for factory in round_record.factories
            ]
            game.setup_round(factories=explicit_factories)

            # Insert initial state before any moves in the first round.
            if not computed_turns:
                computed_turns.append(
                    {
                        "round": round_record.round,
                        "player_index": 0,
                        "source": None,
                        "tile": None,
                        "destination": None,
                        "boards": [
                            _board_to_dict_with_pending(p) for p in game.state.players
                        ],
                        "factories": [
                            [t.name for t in f] for f in game.state.factories
                        ],
                        "center": [t.name for t in game.state.center],
                        "bag_counts": _counts(game.state.bag),
                        "discard_counts": _counts(game.state.discard),
                        "grand_totals": [earned_score(p) for p in game.state.players],
                        "is_initial": True,
                    }
                )

            for move_record in round_record.moves:
                move = Move(
                    source=move_record.source,
                    tile=Tile[move_record.tile],
                    destination=move_record.destination,
                )
                game.make_move(move)

                grand_totals = [earned_score(p) for p in game.state.players]
                computed_turns.append(
                    {
                        "round": round_record.round,
                        "player_index": move_record.player_index,
                        "source": move_record.source,
                        "tile": move_record.tile,
                        "destination": move_record.destination,
                        "boards": [
                            _board_to_dict_with_pending(p) for p in game.state.players
                        ],
                        "factories": [
                            [t.name for t in f] for f in game.state.factories
                        ],
                        "center": [t.name for t in game.state.center],
                        "bag_counts": _counts(game.state.bag),
                        "discard_counts": _counts(game.state.discard),
                        "grand_totals": grand_totals,
                    }
                )

        game.score_game()
        final_boards = [_board_to_dict(p) for p in game.state.players]
        return computed_turns, final_boards

    # ── Serialization ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "game_id": self.game_id,
            "timestamp": self.timestamp,
            "player_names": self.player_names,
            "player_types": self.player_types,
            "rounds": [
                {
                    "round": round_record.round,
                    "factories": round_record.factories,
                    "center": round_record.center,
                    "moves": [
                        {
                            "player_index": move.player_index,
                            "source": move.source,
                            "tile": move.tile,
                            "destination": move.destination,
                        }
                        for move in round_record.moves
                    ],
                }
                for round_record in self.rounds
            ],
            "final_scores": self.final_scores,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameRecord":
        """Deserialize from a dict (e.g. loaded from JSON)."""
        rounds = [
            RoundRecord(
                round=r["round"],
                factories=r["factories"],
                center=r.get("center", ["FIRST_PLAYER"]),
                moves=[
                    MoveRecord(
                        player_index=m["player_index"],
                        source=m["source"],
                        tile=m["tile"],
                        destination=m["destination"],
                    )
                    for m in r.get("moves", [])
                ],
            )
            for r in data.get("rounds", [])
        ]
        return cls(
            game_id=data["game_id"],
            timestamp=data["timestamp"],
            player_names=data["player_names"],
            player_types=data.get("player_types", []),
            rounds=rounds,
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
    """Records an Azul game round by round for later replay and analysis.

    Usage::

        recorder = GameRecorder(player_names=["Alice", "Bob"])
        recorder.start_round(game)          # call at start of each round
        recorder.record_move(move, player_index=0)  # call before make_move
        ...
        recorder.finalize(game)
        recorder.save("recordings/game.json")
    """

    def __init__(
        self,
        player_names: list[str] | None = None,
        player_types: list[str] | None = None,
    ) -> None:
        if player_names is None:
            player_names = [f"Player {i}" for i in range(PLAYERS)]
        if player_types is None:
            player_types = ["human"] * PLAYERS
        self.player_names = player_names
        self.record = GameRecord(
            game_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            player_names=player_names,
            player_types=player_types,
        )
        self._current_round: RoundRecord | None = None

    def start_round(self, game: Game) -> None:
        """Capture the factory layout at the start of a round.

        Must be called after setup_round() so the factories are filled.
        """
        round_record = RoundRecord(
            round=game.state.round,
            factories=[
                [tile.name for tile in factory] for factory in game.state.factories
            ],
            center=[tile.name for tile in game.state.center],
        )
        self.record.rounds.append(round_record)
        self._current_round = round_record

    def record_move(
        self,
        move: Move,
        player_index: int = 0,
    ) -> None:
        """Record a move within the current round.

        Must be called before game.make_move(move).

        Raises RuntimeError if start_round has not been called yet.
        """
        if self._current_round is None:
            raise RuntimeError("start_round must be called before record_move")
        self._current_round.moves.append(
            MoveRecord(
                player_index=player_index,
                source=move.source,
                tile=move.tile.name,
                destination=move.destination,
            )
        )

    def finalize(self, game: Game) -> None:
        """Record final scores and winner. Call after score_game()."""
        self.record.final_scores = [p.score for p in game.state.players]
        scores = self.record.final_scores
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        self.record.winner = winners[0] if len(winners) == 1 else None

    def to_json(self) -> str:
        """Serialize the full game record to a JSON string."""
        return json.dumps(self.record.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        """Write the game record to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
