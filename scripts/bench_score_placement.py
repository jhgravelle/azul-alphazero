"""Quick benchmark: score_placement v1 (direction loop) vs v2 (pointer walk).

Run from project root:
    python scripts/bench_score_placement.py
"""

import timeit
from engine.constants import Tile, BOARD_SIZE, WALL_PATTERN
from engine.scoring import score_placement

# ---------------------------------------------------------------------------
# v2 inline for fair comparison (no import indirection)
# ---------------------------------------------------------------------------


def score_placement_2(wall: list[list[Tile | None]], row: int, col: int) -> int:
    h_start, h_end = col, col
    while h_start - 1 >= 0 and wall[row][h_start - 1] is not None:
        h_start -= 1
    while h_end + 1 < BOARD_SIZE and wall[row][h_end + 1] is not None:
        h_end += 1
    h = h_end - h_start + 1

    v_start, v_end = row, row
    while v_start - 1 >= 0 and wall[v_start - 1][col] is not None:
        v_start -= 1
    while v_end + 1 < BOARD_SIZE and wall[v_end + 1][col] is not None:
        v_end += 1
    v = v_end - v_start + 1

    return (h if h > 1 else 0) + (v if v > 1 else 0) or 1


# ---------------------------------------------------------------------------
# Test cases: lone tile, busy center, full row+col
# ---------------------------------------------------------------------------


def _make_wall() -> list[list[Tile | None]]:
    return [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]


cases = {}

# Lone tile — worst case for early-exit: no neighbours to find
w = _make_wall()
w[2][2] = Tile.RED
cases["lone"] = (w, 2, 2)

# Busy center — + shape, 3 wide 3 tall
w = _make_wall()
for tile, (r, c) in zip(
    [Tile.RED, Tile.BLACK, Tile.YELLOW, Tile.WHITE, Tile.BLUE],
    [(1, 2), (3, 2), (2, 1), (2, 3), (2, 2)],
):
    w[r][c] = tile
cases["center_plus"] = (w, 2, 2)

# Full row + full column through (2,2)
w = _make_wall()
for col in range(BOARD_SIZE):
    w[2][col] = WALL_PATTERN[2][col]
for row in range(BOARD_SIZE):
    w[row][2] = WALL_PATTERN[row][2]
cases["full_cross"] = (w, 2, 2)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

N = 500_000

print(f"{'case':<15} {'v1 (loop)':>12} {'v2 (pointer)':>14} {'winner':>8}")
print("-" * 55)

for name, (wall, row, col) in cases.items():
    t1 = timeit.timeit(lambda: score_placement(wall, row, col), number=N)
    t2 = timeit.timeit(lambda: score_placement_2(wall, row, col), number=N)
    winner = "v1" if t1 < t2 else "v2"
    print(f"{name:<15} {t1:>11.3f}s {t2:>13.3f}s {winner:>8}")

print()
print("Sanity check (results must match):")
for name, (wall, row, col) in cases.items():
    r1 = score_placement(wall, row, col)
    r2 = score_placement_2(wall, row, col)
    status = "OK" if r1 == r2 else f"MISMATCH {r1} vs {r2}"
    print(f"  {name:<15} v1={r1}  v2={r2}  {status}")
