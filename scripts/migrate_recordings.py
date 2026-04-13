"""Migrate existing recording files to the current format.

Safe to run multiple times -- files already at the current schema are
left unchanged. A .bak file is written alongside each migrated file so
the original is never lost.

Usage:
    python -m scripts.migrate_recordings
    python -m scripts.migrate_recordings --recordings-dir path/to/recordings
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Schema defaults ────────────────────────────────────────────────────────
# Add a new entry here whenever a field is added to GameRecord or TurnRecord.
# Key: field name. Value: default to fill in when the field is absent.

GAME_RECORD_DEFAULTS: dict = {
    "player_types": [],
}

TURN_DEFAULTS: dict = {
    "round": 0,
    "grand_totals": [],
    "source_state": {
        "factories": [],
        "center": [],
        "bag_counts": {},
        "discard_counts": {},
    },
    "analysis": None,
}


def migrate_record(data: dict) -> tuple[dict, bool]:
    """Return (migrated_data, was_changed).

    Fills in any missing fields with their defaults. Does not remove
    unknown fields -- forward compatibility is not our concern here.
    """
    changed = False

    for key, default in GAME_RECORD_DEFAULTS.items():
        if key not in data:
            data[key] = default
            changed = True
            logger.info("  added game field: %s = %r", key, default)

    for turn in data.get("turns", []):
        for key, default in TURN_DEFAULTS.items():
            if key not in turn:
                turn[key] = default
                changed = True

    return data, changed


def migrate_directory(recordings_dir: Path, dry_run: bool = False) -> None:
    """Migrate all JSON files in recordings_dir."""
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
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("SKIP %s -- could not parse: %s", path.name, exc)
            continue

        data, changed = migrate_record(data)

        if not changed:
            logger.info("OK   %s", path.name)
            skipped += 1
            continue

        if dry_run:
            logger.info("DRY  %s -- would update", path.name)
            migrated += 1
            continue

        # Write backup alongside original before overwriting.
        backup = path.with_suffix(".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("MIGRATED %s (backup -> %s)", path.name, backup.name)
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
