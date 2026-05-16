# engine/game_recorder.py
"""Game recorder for capturing Azul games for replay and analysis."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.constants import (
    BONUS_COLUMN,
    BONUS_ROW,
    BONUS_TILE,
    CELLS_BY_COL,
    CELLS_BY_ROW,
    CELLS_BY_TILE,
    COL_FOR_TILE_ROW,
    PLAYERS,
)
from engine.game import Game, Move, Tile
from engine.player import Player

# ── Data classes ───────────────────────────────────────────────────────────


@dataclass
class MoveRecord:
    """A single move within a round."""

    player_index: int
    source: int
    tile: str
    destination: int


@dataclass
class TurnRecord:
    """A single turn: move and resulting game state."""

    turn: int
    state: list[str]
    move: str


@dataclass
class RoundRecord:
    """The factory layout and moves for one round."""

    round: int
    factories: list[list[str]] = field(default_factory=list)
    center: list[str] = field(default_factory=list)
    round_display: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    moves: list[MoveRecord] = field(default_factory=list)


# ── Pending breakdown helpers ──────────────────────────────────────────────


def _pending_placement_details(player: Player) -> list[dict[str, Any]]:
    """Return placement details for all full pattern lines.

    Simulates end-of-round placement in row order on a temporary wall copy
    so adjacency scores are correct when pending placements are adjacent.
    """
    from engine.constants import CAPACITY

    wall = [[1 if player.wall[row][col] else 0 for col in range(5)] for row in range(5)]
    details = []
    for row in range(5):
        pattern_row = player.pattern_lines[row]
        if not pattern_row:
            continue
        tile = pattern_row[0]
        col = COL_FOR_TILE_ROW[tile][row]
        if len(pattern_row) < CAPACITY[row]:
            continue
        wall[row][col] = 1
        points = _score_placement(wall, row, col)
        details.append({"row": row, "column": col, "placement_points": points})
    return details


def _score_placement(wall: list[list[int]], row: int, col: int) -> int:
    """Score a single tile at (row, col) on the wall.

    Precondition: wall[row][col] must already be set before calling.
    """
    from engine.constants import SIZE

    h_start, h_end = col, col
    while h_start - 1 >= 0 and wall[row][h_start - 1]:
        h_start -= 1
    while h_end + 1 < SIZE and wall[row][h_end + 1]:
        h_end += 1
    h = h_end - h_start + 1

    v_start, v_end = row, row
    while v_start - 1 >= 0 and wall[v_start - 1][col]:
        v_start -= 1
    while v_end + 1 < SIZE and wall[v_end + 1][col]:
        v_end += 1
    v = v_end - v_start + 1

    return (h if h > 1 else 0) + (v if v > 1 else 0) or 1


def _pending_bonus_details(
    wall: list[list[int]],
) -> list[dict[str, Any]]:
    """Return bonus details for all end-of-game bonuses guaranteed by the wall."""
    bonuses = []
    for row_idx, cells in enumerate(CELLS_BY_ROW):
        if all(wall[r][c] for r, c in cells):
            bonuses.append(
                {"bonus_type": "row", "index": row_idx, "bonus_points": BONUS_ROW}
            )
    for col_idx, cells in enumerate(CELLS_BY_COL):
        if all(wall[r][c] for r, c in cells):
            bonuses.append(
                {"bonus_type": "column", "index": col_idx, "bonus_points": BONUS_COLUMN}
            )
    for tile_idx, cells in enumerate(CELLS_BY_TILE):
        if all(wall[r][c] for r, c in cells):
            bonuses.append(
                {"bonus_type": "tile", "index": tile_idx, "bonus_points": BONUS_TILE}
            )
    return bonuses


def _build_post_placement_wall(player: Player) -> list[list[int]]:
    """Return a copy of the wall with all pending full pattern lines placed."""
    from engine.constants import CAPACITY

    wall = [[1 if player.wall[row][col] else 0 for col in range(5)] for row in range(5)]
    for row in range(5):
        pattern_row = player.pattern_lines[row]
        if not pattern_row:
            continue
        tile = pattern_row[0]
        col = COL_FOR_TILE_ROW[tile][row]
        if len(pattern_row) >= CAPACITY[row]:
            wall[row][col] = 1
    return wall


# ── Board serialization ────────────────────────────────────────────────────


def _player_to_dict(player: Player) -> dict[str, Any]:
    """Serialize a Player to a JSON-compatible dict."""

    pattern_lines = []
    for row in range(5):
        pattern_row = player.pattern_lines[row]
        if not pattern_row:
            pattern_lines.append([])
        else:
            tile = pattern_row[0]
            pattern_lines.append([tile.name] * len(pattern_row))
    wall = [
        [tile.name if tile else None for tile in player.wall[row]] for row in range(5)
    ]
    return {
        "score": player.score,
        "pattern_lines": pattern_lines,
        "wall": wall,
        "floor_line": [tile.name for tile in player.floor_line],
    }


def _player_to_dict_with_pending(player: Player) -> dict[str, Any]:
    """Serialize a player including pending placement and bonus details."""
    post_wall = _build_post_placement_wall(player)
    placement_details = _pending_placement_details(player)
    bonus_details = _pending_bonus_details(post_wall)
    base = _player_to_dict(player)
    base["pending_placements"] = placement_details
    base["pending_bonuses"] = bonus_details
    return base


def _counts(tile_list: list[Tile]) -> dict[str, int]:
    colors = [t for t in Tile if t != Tile.FIRST_PLAYER]
    return {t.name: tile_list.count(t) for t in colors}


# ── GameRecord ─────────────────────────────────────────────────────────────


@dataclass
class GameRecord:
    """The complete record of one Azul game."""

    game_id: str
    timestamp: str
    player_names: list[str]
    player_types: list[str] = field(default_factory=list)
    rounds: list[RoundRecord] = field(default_factory=list)
    final_scores: list[int] = field(default_factory=list)
    winner: int | None = None
    final_score_display: str = ""
    final_state: list[str] = field(default_factory=list)

    def reconstruct(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Replay all moves and return (computed_turns, final_boards)."""
        game = Game()
        computed_turns: list[dict[str, Any]] = []

        for round_record in self.rounds:
            explicit_factories = [
                [Tile[name] for name in factory] for factory in round_record.factories
            ]
            game.setup_round(factories=explicit_factories)

            if not computed_turns:
                computed_turns.append(
                    {
                        "round": round_record.round,
                        "player_index": 0,
                        "source": None,
                        "tile": None,
                        "destination": None,
                        "boards": [
                            _player_to_dict_with_pending(p) for p in game.players
                        ],
                        "factories": [[t.name for t in f] for f in game.factories],
                        "center": [t.name for t in game.center],
                        "bag_counts": _counts(game.bag),
                        "discard_counts": _counts(game.discard),
                        "grand_totals": [p.earned for p in game.players],
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
                game.advance(skip_setup=True)

                computed_turns.append(
                    {
                        "round": round_record.round,
                        "player_index": move_record.player_index,
                        "source": move_record.source,
                        "tile": move_record.tile,
                        "destination": move_record.destination,
                        "boards": [
                            _player_to_dict_with_pending(p) for p in game.players
                        ],
                        "factories": [[t.name for t in f] for f in game.factories],
                        "center": [t.name for t in game.center],
                        "bag_counts": _counts(game.bag),
                        "discard_counts": _counts(game.discard),
                        "grand_totals": [p.earned for p in game.players],
                    }
                )

        if game.is_game_over():
            game._score_game()
        final_boards = [_player_to_dict(p) for p in game.players]
        return computed_turns, final_boards

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "timestamp": self.timestamp,
            "rounds": [
                {
                    "round": r.round,
                    "round_display": r.round_display,
                    "turns": [
                        {
                            "turn": t.turn,
                            "state": t.state,
                            "move": t.move,
                        }
                        for t in r.turns
                    ],
                }
                for r in self.rounds
            ],
            "final_score_display": self.final_score_display,
            "final_state": self.final_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameRecord":
        """Deserialize from a dict (e.g. loaded from JSON).

        Note: reconstructing games from JSON with only state strings (no factories)
        requires parsing the game display. For now, we populate factories field
        from data if available, but it's optional for new JSON format.
        """
        rounds = [
            RoundRecord(
                round=r["round"],
                factories=r.get("factories", []),
                center=r.get("center", ["FIRST_PLAYER"]),
                round_display=r.get("round_display", ""),
                turns=[
                    TurnRecord(
                        turn=t["turn"],
                        state=(
                            t.get("state", [])
                            if isinstance(t.get("state"), list)
                            else [t.get("state", "")]
                        ),
                        move=t["move"],
                    )
                    for t in r.get("turns", [])
                ],
            )
            for r in data.get("rounds", [])
        ]
        final_state = data.get("final_state", [])
        if isinstance(final_state, str):
            final_state = [final_state] if final_state else []

        return cls(
            game_id=data.get("game_id", ""),
            timestamp=data["timestamp"],
            player_names=data.get("player_names", []),
            player_types=data.get("player_types", []),
            rounds=rounds,
            final_scores=data.get("final_scores", []),
            winner=data.get("winner"),
            final_score_display=data.get("final_score_display", ""),
            final_state=final_state,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GameRecord":
        """Load a GameRecord from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ── Recorder ───────────────────────────────────────────────────────────────


class GameRecorder:
    """Records an Azul game round by round for later replay and analysis."""

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
        self._turn_count: int = 0

    def start_round(self, game: Game) -> None:
        """Capture the factory layout at the start of a round."""
        round_display = f"=== Round {game.round} ==="
        round_record = RoundRecord(
            round=game.round,
            round_display=round_display,
            factories=[[tile.name for tile in factory] for factory in game.factories],
            center=[tile.name for tile in game.center],
        )
        self.record.rounds.append(round_record)
        self._current_round = round_record
        self._turn_count = 0

    def record_move(self, move: Move, game: Game, player_index: int = 0) -> None:
        """Record a move within the current round."""
        if self._current_round is None:
            raise RuntimeError("start_round must be called before record_move")
        self._turn_count += 1
        move_str = self._format_move(move, game)
        state_lines = str(game).splitlines()
        turn_record = TurnRecord(
            turn=self._turn_count,
            state=state_lines,
            move=move_str,
        )
        self._current_round.turns.append(turn_record)
        self._current_round.moves.append(
            MoveRecord(
                player_index=player_index,
                source=move.source,
                tile=move.tile.name,
                destination=move.destination,
            )
        )

    def _format_move(self, move: Move, game: Game) -> str:
        """Format a move as a string with count from tiles in the source."""
        from engine.constants import CENTER

        if move.source == CENTER:
            tiles = game.center
        else:
            tiles = game.factories[move.source]
        count = sum(1 for t in tiles if t == move.tile)
        took_first = move.source == CENTER and Tile.FIRST_PLAYER in tiles

        move_with_count = Move(
            tile=move.tile,
            source=move.source,
            destination=move.destination,
            count=count,
            took_first=took_first,
        )
        return str(move_with_count)

    def finalize(self, game: Game) -> None:
        """Record final scores and winner."""
        self.record.final_scores = [p.score for p in game.players]
        scores = self.record.final_scores
        best = max(scores)
        winners = [i for i, s in enumerate(scores) if s == best]
        self.record.winner = winners[0] if len(winners) == 1 else None

        self.record.final_state = str(game).splitlines()
        score_parts = [
            f"{name}: {score}" for name, score in zip(self.player_names, scores)
        ]
        self.record.final_score_display = "  ".join(score_parts)

    def to_json(self) -> str:
        """Serialize the full game record to a JSON string."""
        return json.dumps(self.record.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        """Write the game record to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
