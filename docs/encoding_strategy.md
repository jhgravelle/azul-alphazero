# Encoding Strategy — 168-Value Player State

---

## Overview

**Goal:** Encode a player's board state into 168 fixed-size values for supervised learning. The encoding must capture all information needed for the value network to predict final game outcome.

**Constraints:**
- Fixed size (no variable-length sequences)
- Deterministic (same board state → same encoding always)
- Interpretable (future reference: what does each section mean?)
- Complete (nothing important omitted)

**Layout:** 10 sections, 168 values total

---

## The 168-Value Layout

### Section 1: Wall Cells (0–24, 25 values)
**Binary: has this color-cell been placed on the wall?**

| Indices | Count | Content |
|---|---|---|
| 0–24 | 25 | Wall cells (5 colors × 5 rows, row-major) |

**Interpretation:**
- Index = `color * 5 + row`
- Value = 0 (not placed) or 1 (placed)
- Deterministic wall state (ignores pending tiles)

**Why:** Network needs to know what's completed to compute adjacency bonuses and avoid placing duplicate colors.

---

### Section 2: Pending Wall (25–49, 25 values)
**Binary: is this cell pending (pattern line full, will place next round)?**

| Indices | Count | Content |
|---|---|---|
| 25–49 | 25 | Pending wall cells |

**Interpretation:**
- Index = `color * 5 + row`
- Value = 0 (not pending) or 1 (pending placement)
- Only 1 if pattern line is full and committed to this color

**Why:** Pending cells block placements but don't score yet. Value net must see them to avoid planning on locked rows.

---

### Section 3: Adjacency Grid (50–74, 25 values)
**Integer: length of longest contiguous run of tiles (horizontal + vertical) touching this cell.**

| Indices | Count | Content |
|---|---|---|
| 50–74 | 25 | Adjacency run count per cell (0–10) |

**Interpretation:**
- Index = `color * 5 + row`
- Value = contiguous run count including this cell
  - 0: cell empty (no tiles on this color's row)
  - 1: isolated tile (no adjacent tiles same color)
  - 2–10: run of 2–10 contiguous tiles
- Max value = 10 (full row is 5-wide per the wall, but diagonals can extend beyond)

**Why:** Adjacency directly affects scoring. Placing a tile in position (row, col) gets bonus for each adjacent placed tile. Network needs this to evaluate score-now vs setup-for-later tradeoffs.

---

### Section 4: Wall Row Demand (75–99, 25 values)
**Integer: per color per row, how many tiles still needed to complete?**

| Indices | Count | Content |
|---|---|---|
| 75–99 | 25 | Wall row demand (5 colors × 5 rows) |

**Interpretation:**
- Index = `color * 5 + row`
- Value = remaining capacity for this (color, row)
  - Range: 0–CAPACITY[row]
  - CAPACITY = [1, 2, 3, 4, 5] (row 0 has 1 cell, row 4 has 5 cells)
  - 0 = row complete, no more tiles needed
  - CAPACITY[row] = row empty, need all tiles

**Why:** Network needs to know "which rows are close to completion" to prioritize patterns and evaluate lock-in risk.

---

### Section 5: Wall Column Demand (100–124, 25 values)
**Integer per color per column, how many tiles needed — sorted by completion order.**

| Indices | Count | Content |
|---|---|---|
| 100–124 | 25 | Wall column demand (5 colors × 5 cols, sorted) |

**Interpretation:**
- Grid is 5 colors × 5 cols
- Values sorted ascending by total demand per color (most complete color first)
- Within each color, cols sorted by total demand
- Value = tiles needed in that (color, col)
  - Range: 0–5 (each col can have 0–5 tiles across rows)

**Why:** Columns give col-completion bonuses (10 pts). Network benefits from seeing which colors are closest to col completion, prioritized by overall progress.

---

### Section 6: Wall Tile Demand (125–129, 5 values)
**Integer per color: total tiles needed across entire wall.**

| Indices | Count | Content |
|---|---|---|
| 125–129 | 5 | Total wall tiles needed per color |

**Interpretation:**
- Index = color (0–4)
- Value = total cells still needed for that color
  - Range: 0–25 (5 rows × 5 cells per color)
  - 0 = all 25 cells placed
  - 25 = no cells placed

**Why:** Summarizes wall completion state per color. Network can quickly see "which colors are locked in vs which need investment."

---

### Section 7: Pattern Demand (130–134, 5 values)
**Integer per color: total tiles needed to complete started pattern lines.**

| Indices | Count | Content |
|---|---|---|
| 130–134 | 5 | Pattern line tiles needed per color |

**Interpretation:**
- Index = color (0–4)
- Value = tiles needed across all committed (full) pattern lines for this color
  - Only counts full lines (lines that will place end-of-round)
  - Range: 0–5 (at most 5 full lines × 1 tile needed per line)

**Why:** Distinguishes between "pattern is active" (will place) vs "pattern is dormant" (incomplete). Network needs to know which lines are committed.

---

### Section 8: Pattern Capacity (135–159, 25 values)
**Integer per color per row: how many tiles fit in the pattern line this round?**

| Indices | Count | Content |
|---|---|---|
| 135–159 | 25 | Pattern line capacity per color per row |

**Interpretation:**
- Index = `color * 5 + row`
- Value = tiles that fit in this pattern line this round (0–CAPACITY[row])
  - 0 = pattern line full (committed or started)
  - CAPACITY[row] = pattern line empty
  - CAPACITY = [1, 2, 3, 4, 5]

**Why:** Network needs to know "can I safely place 3 tiles of this color this round?" to evaluate move legality and scoring impact.

---

### Section 9: Scoring (160–164, 5 values)
**Integer scores in different states.**

| Indices | Count | Content |
|---|---|---|
| 160 | 1 | Official score (settled, from prior rounds) |
| 161 | 1 | Pending score (full pattern lines, pending placement) |
| 162 | 1 | Penalty (floor line) |
| 163 | 1 | Bonus (adjacency + row/col/tile bonuses) |
| 164 | 1 | Earned (official + pending + penalty + bonus) |

**Interpretation:**
- All raw int values (no normalization in encoding)
- Divisors applied by model as needed
- Range: typically −14 to +100 unclamped
- Earned = official + pending + penalty + bonus

**Why:** Value network must predict final score differential. Seeing current score state (split into components) helps network learn which scoring paths lead to victory.

---

### Section 10: Misc (165–167, 3 values)
**Special flags and counts.**

| Indices | Count | Content |
|---|---|---|
| 165 | 1 | First player token in floor (0 or 1) |
| 166 | 1 | Total tiles used (placed on wall) |
| 167 | 1 | Max pattern capacity (max of all color rows) |

**Interpretation:**
- Token: 0 = no, 1 = yes
- Used: sum of wall cells placed (0–25)
- Max capacity: max value across all 25 pattern cells (0–5)

**Why:** First player token signals who goes next. Used tiles summarize progress. Max capacity tells network "how much can I do this turn max?"

---

## Encoding Process

### When Encoding Happens
Encoding is computed on-demand:
- After `player.place()` (tile placed on pattern/floor)
- After `player.process_round_end()` (pattern→wall transfer, scoring)
- When reading `player.encoded_features` property

### Caching Strategy
`encoded_features` is a mutable list stored on player object. Updated whenever board state changes, so reads are O(1).

### Code Structure
```python
class Player:
    def _encode(self):
        """Recompute all 168 values."""
        wall = [... 25 values ...]
        pending = [... 25 values ...]
        adjacency = [... 25 values ...]
        # ... etc ...
        self.encoded_features = flatten([wall, pending, adjacency, ...])
    
    def place(self, ...):
        # ... place logic ...
        self._encode()  # update cache
    
    @property
    def encoded_features(self):
        return self._cached_features
```

---

## ENCODING_SLICES Reference

Fixed slices for reliable indexing:

```python
ENCODING_SLICES = {
    "wall": slice(0, 25),
    "pending_wall": slice(25, 50),
    "adjacency_grid": slice(50, 75),
    "wall_row_demand": slice(75, 100),
    "wall_col_demand": slice(100, 125),
    "wall_tile_demand": slice(125, 130),
    "pattern_demand": slice(130, 135),
    "pattern_capacity": slice(135, 160),
    "scoring": slice(160, 165),
    "misc": slice(165, 168),
}
```

Use slices to extract sections:
```python
wall_section = player.encoded_features[ENCODING_SLICES["wall"]]
```

---

## Normalization & Model Integration

The 168 raw values feed into the neural network. The model applies divisors:

| Section | Divisor | Reason |
|---|---|---|
| Wall, Pending | 1.0 | Binary already [0,1] |
| Adjacency | 10.0 | Range [0,10] |
| Demand/Capacity | 5.0 | Range [0,5] |
| Scores | 100.0 (absolute), 50.0 (diff) | Unbounded → normalized |
| Misc | 1.0, 25.0, 5.0 | Per-field divisor |

The model uses an `ENCODING_DIVISORS` dict to apply these during forward pass:
```python
normalized = [raw / divisor for raw, divisor in zip(encoding, ENCODING_DIVISORS)]
```

---

## Redundant Features (Considered & Rejected)

These features were evaluated and found redundant or harmful. **If we reconsider, start here.**

### 1. Raw Factory Distribution
**Feature:** Tile counts remaining in center and factories (e.g., `[10, 8, 9, 7, 6]` per color)

**Why rejected:**
- Highly variable (depends on history this round)
- Not predictive of end-of-game value
- Round boundary constraint prevents seeing next-round factories anyway
- Including this would encourage overfitting to factory state instead of board state

**Reconsider if:** We move beyond round boundaries and want to predict "luck" from factory distribution.

---

### 2. Per-Cell Wall Strength (Diagonal Neighbors)
**Feature:** For each wall cell, count diagonal + horizontal neighbors

**Why rejected:**
- Adjacency grid (Section 3) already captures this
- Adding "diagonal strength" is redundant when adjacency is present
- Would bloat encoding without new information

**Reconsider if:** Model struggles with complex adjacency patterns and we want explicit corner-bonuses encoded.

---

### 3. Pattern Line Commitment Flags
**Feature:** Binary per line: "will this line place next round?"

**Why rejected:**
- Pattern demand (Section 7) already encodes this: if demand > 0, it's committed
- Redundant with capacity (Section 8): if capacity = 0, it's full and committed
- Would add 5 more values for no new information

**Reconsider if:** Pattern commitment logic becomes complex and we want explicit flags.

---

### 4. Opponent Encoded State
**Feature:** Mirror of the 168 values, but for opponent

**Why rejected:**
- Would double encoding size to 336
- Many features (wall, pattern) are symmetric, no new information
- Scoring diffs capture what matters (my score − opponent score)
- For two-player game, can compute opponent impact via relative scoring

**Reconsider if:** Opponent state becomes critical (e.g., detecting threat patterns on their wall).

---

### 5. Bag Tile Counts
**Feature:** Count of each color remaining in bag

**Why rejected:**
- Round boundary constraint: bag state changes when round resets
- Not accessible at round boundary anyway (would need to track outside game state)
- Next round's factory depends on bag, but we don't predict cross-round

**Reconsider if:** We extend encoding to handle mid-round decisions (not round-boundary-only).

---

### 6. Move History (Tiles Taken This Round)
**Feature:** Sequence of moves made so far

**Why rejected:**
- Redundant with current board state (move history is encoded in wall/pattern)
- Sequence would break fixed-size encoding requirement
- Order-dependent features don't generalize well in supervised learning

**Reconsider if:** We model order-dependent patterns (e.g., "which color did I prioritize?").

---

### 7. Turn Counter (Which Player's Turn)
**Feature:** 0 or 1 indicating whose turn it is

**Why rejected:**
- All encoded features are **player-local** (my wall, my pattern, my score)
- Turn is implicit in the game state (already known when reading encoding)
- For network, we evaluate positions from the moving player's perspective

**Reconsider if:** We model global game state instead of player-local encodings.

---

### 8. Tile Availability Filtered by Row
**Feature:** For each wall row, count tiles available in each color

**Why rejected:**
- Tile availability changes constantly (depends on moves)
- Round boundary: we don't look ahead to next round
- Wall row/col demand (Sections 4–5) better capture "which rows are contested"

**Reconsider if:** We find the model can't learn row-specific placement rules from wall demand alone.

---

## Design Philosophy

**Why 168 values?**

The number wasn't arbitrary. Each section was added to answer a specific question the value network must answer:

- **Wall + Pending** — "What's locked in vs what can still happen?"
- **Adjacency** — "What's the immediate scoring for placements?"
- **Demand grids** — "Which colors/rows are close to bonuses?"
- **Pattern/Capacity** — "What moves are legal? What scores?"
- **Scoring breakdown** — "Who's winning by how much, and why?"
- **Misc** — "Are there special conditions (first player)?"

Together, these 168 values encode everything needed to predict "will this position lead to a win?"

---

## Testing Strategy

All sections are validated in `tests/engine/test_player.py`:
- Shape tests (exactly 168 values, all ints)
- Slice tests (no gaps, complete coverage)
- Semantic tests (wall cells match actual placements, demand matches actual remaining, etc.)
- Consistency tests (idempotency, updates after board changes)

---

## References

- [[master_plan.md]] — Vision and phase overview
- [[alphabeta_strategy.md]] — How AlphaBeta trains value nets
- [[phase_0_refactor.md]] — Current encoding refactoring work

---

**Status:** Encoding complete and documented. Tests in progress. Ready for downstream refactoring once tests pass.
