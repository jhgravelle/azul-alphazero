# engine/game_recorder.py
"""Game recorder for capturing Azul games for replay and analysis."""

import json
import re
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
    """The starting state and turns for one round."""

    round: int
    round_display: str = ""
    starting_state: list[str] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)


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


def _extract_seed_from_starting_state(starting_state: list[str]) -> int:
    """Extract the seed from the first line of a starting state.

    Format: R{round}:T{turn} [{seed:010d}]
    Uses regex to extract the seed value in square brackets.

    Args:
        starting_state: List of state display strings, first line has header

    Returns:
        The seed as an integer

    Raises:
        ValueError: If the seed cannot be extracted
    """
    if not starting_state:
        raise ValueError("starting_state cannot be empty")
    first_line = starting_state[0]
    match = re.search(r"\[(\d+)\]", first_line)
    if not match:
        raise ValueError(f"Could not extract seed from: {first_line!r}")
    return int(match.group(1))


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
        """Replay all moves and return (computed_turns, final_boards).

        Creates a single Game instance with the seed from the first round,
        then replays all moves sequentially across all rounds. Uses game.advance()
        to handle round transitions automatically.
        """
        computed_turns: list[dict[str, Any]] = []

        if not self.rounds:
            return computed_turns, []

        # Extract seed from the first round's starting state
        seed = _extract_seed_from_starting_state(self.rounds[0].starting_state)

        # Create a single game instance with the seed
        game = Game(seed=seed)

        # Track whether this is the first entry
        is_first_entry = True

        # Replay all moves across all rounds
        for round_record in self.rounds:
            # Initialize the round (on first round, call setup_round; on later rounds,
            # it's already been set up by advance() at the previous round boundary)
            if round_record.round == 1:
                game.setup_round()

            # Add initial state for this round (only on first round)
            if is_first_entry:
                computed_turns.append(
                    {
                        "round": round_record.round,
                        "player_index": game.current_player_index,
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
                is_first_entry = False

            # Replay all turns in the round
            for turn_record in round_record.turns:
                move = None
                if turn_record.move:
                    move = Move.from_str(turn_record.move)
                    game.make_move(move)
                    game.advance()

                computed_turns.append(
                    {
                        "round": round_record.round,
                        "player_index": game.current_player_index,
                        "source": move.source if move else None,
                        "tile": move.tile.name if move else None,
                        "destination": move.destination if move else None,
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
            "game_id": self.game_id,
            "timestamp": self.timestamp,
            "rounds": [
                {
                    "round": r.round,
                    "round_display": r.round_display,
                    "starting_state": r.starting_state,
                    "turns": [
                        {
                            "turn": t.turn,
                            "move": t.move,
                            "state": t.state,
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

        Expects the new format with starting_state for each round.
        """
        rounds = [
            RoundRecord(
                round=r["round"],
                round_display=r.get("round_display", ""),
                starting_state=r.get("starting_state", []),
                turns=[
                    TurnRecord(
                        turn=t["turn"],
                        move=t["move"],
                        state=(
                            t.get("state", [])
                            if isinstance(t.get("state"), list)
                            else [t.get("state", "")]
                        ),
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
        """Capture the game state at the start of a round."""
        round_display = f"=== Round {game.round} ==="
        round_record = RoundRecord(
            round=game.round,
            round_display=round_display,
        )
        round_record.starting_state = str(game).splitlines()
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
            move=move_str,
            state=state_lines,
        )
        self._current_round.turns.append(turn_record)

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
