#!/usr/bin/env python
# scripts/parse_log.py

"""Extract iteration summaries from a training log file.

Usage:
    python -m scripts.parse_log logs/run_20260404_235616.log
    python -m scripts.parse_log logs/run_20260404_235616.log --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Plain substrings to search for in each line — simpler and more reliable
# than regex when the log format is fixed.
_KEYWORDS = [
    "run parameters:",
    "using device:",
    "heuristic summary",
    "pretraining complete",
    "heuristic iterations complete",
    "switching to self-play mode",
    "iter  ",  # per-iteration summary lines e.g. "iter   1 |"
    "iter   ",
    "Training Summary",
    "total generations:",
    "best checkpoint:",
    "warmup enabled",
]

_SEPARATORS = ("----", "====")


def matches(line: str) -> bool:
    for kw in _KEYWORDS:
        if kw in line:
            return True
    for sep in _SEPARATORS:
        if sep in line and len(line.strip()) > 20:
            return False  # skip pure separator lines in normal output
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract iteration summaries from a training log"
    )
    parser.add_argument("log_file", type=Path, help="path to the log file")
    parser.add_argument(
        "--all",
        action="store_true",
        help="include all INFO lines, not just summaries",
    )
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"error: file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    lines_read = 0
    lines_printed = 0

    with args.log_file.open(encoding="utf-8") as f:
        for line in f:
            lines_read += 1
            line = line.rstrip()
            if args.all:
                if "INFO" in line or "WARNING" in line:
                    print(line)
                    lines_printed += 1
            elif matches(line):
                print(line)
                lines_printed += 1

    if lines_printed == 0:
        print(
            f"no matching lines found in {args.log_file} " f"({lines_read} lines read)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
