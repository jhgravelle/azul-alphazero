"""Migrate existing recording files to the current compact round/move format.

Old format: flat list of turns, each with full board_states and source_state.
New format: list of rounds, each with factory layout and list of moves.

Safe to run multiple times -- already-migrated files are detected and skipped.
A .bak file is written alongside each migrated file.

Usage:
    python -m scripts.migrate_recordings
    python -m scripts.migrate_recordings --dry-run
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _is_old_format(data: dict) -> bool:
    """Return True if this recording uses the old flat-turns format."""
    return "turns" in data and "rounds" not in data


def _is_round_boundary(prev_turn: dict | None, curr_turn: dict) -> bool:
    """Return True if curr_turn starts a new round.

    Detects boundary by checking if previous turn's factories were all empty
    and current turn's factories are filled (new round was set up between them).
    Also triggers on the very first turn.
    """
    if prev_turn is None:
        return True
    prev_factories = prev_turn.get("source_state", {}).get("factories", [])
    curr_factories = curr_turn.get("source_state", {}).get("factories", [])
    prev_empty = all(len(f) == 0 for f in prev_factories)
    curr_has_tiles = any(len(f) > 0 for f in curr_factories)
    return prev_empty and curr_has_tiles


def _migrate_old_to_new(data: dict) -> dict:
    turns = data.get("turns", [])
    rounds: list[dict] = []
    current_round: dict | None = None
    round_number = 0

    # Prefix player names with P1, P2 if not already prefixed.
    player_names = data.get("player_names", [])
    prefixed_names = [
        name if name.startswith(("P1 ", "P2 ", "P3 ", "P4 ")) else f"P{i+1} {name}"
        for i, name in enumerate(player_names)
    ]

    for i, turn in enumerate(turns):
        prev_turn = turns[i - 1] if i > 0 else None
        if _is_round_boundary(prev_turn, turn):
            round_number += 1
            source = turn.get("source_state", {})
            current_round = {
                "round": round_number,
                "factories": source.get("factories", []),
                "center": source.get("center", ["FIRST_PLAYER"]),
                "moves": [],
            }
            rounds.append(current_round)

        if current_round is not None:
            current_round["moves"].append(
                {
                    "player_index": turn["player_index"],
                    "source": turn["move_source"],
                    "tile": turn["move_tile"],
                    "destination": turn["move_destination"],
                }
            )

    return {
        "game_id": data["game_id"],
        "timestamp": data["timestamp"],
        "player_names": prefixed_names,
        "player_types": data.get("player_types", []),
        "rounds": rounds,
        "final_scores": data.get("final_scores", []),
        "winner": data.get("winner"),
    }


def migrate_directory(recordings_dir: Path, dry_run: bool = False) -> None:
    if not recordings_dir.exists():
        logger.info("Directory %s does not exist -- nothing to do.", recordings_dir)
        return

    paths = sorted(recordings_dir.glob("*.json"))
    if not paths:
        logger.info("No recordings found in %s.", recordings_dir)
        return

    logger.info("Found %d recording(s) in %s.", len(paths), recordings_dir)
    migrated = 0
    skipped = 0

    for path in paths:
        if path.suffix == ".bak":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("SKIP %s -- could not parse: %s", path.name, exc)
            continue

        if not _is_old_format(data):
            logger.info("OK   %s", path.name)
            skipped += 1
            continue

        new_data = _migrate_old_to_new(data)

        if dry_run:
            logger.info(
                "DRY  %s -- would migrate %d turns across %d rounds",
                path.name,
                len(data.get("turns", [])),
                len(new_data["rounds"]),
            )
            migrated += 1
            continue

        backup = path.with_suffix(".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(
            json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "MIGRATED %s (%d turns -> %d rounds, backup -> %s)",
            path.name,
            len(data.get("turns", [])),
            len(new_data["rounds"]),
            backup.name,
        )
        migrated += 1

    logger.info("Done -- %d migrated, %d already current.", migrated, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=Path("recordings"),
        help="Path to the recordings directory (default: recordings/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing any files.",
    )
    args = parser.parse_args()
    migrate_directory(args.recordings_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
