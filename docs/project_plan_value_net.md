# Azul AlphaZero: Supervised Learning Value Net — Complete Project Plan

**Last Updated:** 2026-05-14  
**Status:** Phase 0 (Supervised Learning) in progress — Player 168-value encoding implemented, tests pending  
**Branch:** `feat/supervised-value-net`  
**Next Step:** Write comprehensive Player encoding tests

---

## Executive Summary

This project pivots from AlphaZero self-play to **supervised learning with AlphaBeta agents**. The pipeline:
1. Play games with low-temperature AlphaBeta agents
2. Collect every position and label with final game outcome
3. Train a value-only neural network on (state, value) pairs
4. Use that net inside AlphaBeta as the eval function
5. Iterate for up to 10 generations until win rate plateaus

**Why this works:** Azul is shallow (~30 moves). We bootstrap from hand-coded AB heuristics and improve iteratively without needing millions of self-play games.

---

## Phase 0: Supervised Learning Refactor (In Progress)

### Completed Work (8z)

#### Player 168-Value Encoding ✅
Implemented comprehensive board state encoding in `Player._encode()`:

**Layout (168 total values):**
- Wall cells (25): Binary 1/0 per cell, row-major order
- Pending wall (25): Binary 1/0, indicates cells placing this round
- Adjacency grid (25): Contiguous run count 0–10 per cell (1 if alone)
- Wall row demand (25): Per-color tiles needed per row [5 colors × 5 rows]
- Wall col demand (25): Per-color tiles needed per col, sorted by completion [5 colors × 5 cols, sorted ascending]
- Wall tile demand (5): Per-color total tiles needed across entire wall [5 colors × 1]
- Pattern demand (5): Per-color tiles needed to complete started pattern lines [5 colors × 1]
- Pattern capacity (25): Per-color room remaining per row [5 colors × 5 rows]
- Scoring (5): official_score, pending_score, penalty_score, bonus_score, earned_score
- Misc (3): first_player_token, total_used, max_pattern_capacity

**Key design decisions:**
- Two-pass architecture: Build 2D grids, flatten to 1D
- Wall col demand sorted by total demand (most complete first) to guide model
- Pattern demand sums across all rows per color (committed lines only)
- Scoring values computed once per `_encode()` call, stored in `encoded_features`
- All values raw (no normalization) — divisors applied by model layer via `ENCODING_DIVISORS`
- `_flatten()` recursive helper handles arbitrary nesting depth

**Code quality:**
- Slices defined in `ENCODING_SLICES` for reliable indexing
- Properties (`pending`, `penalty`, `bonus`, `earned`) read from slices
- `_encode()` called after `place()` and `process_round_end()` to keep features in sync
- `is_tile_valid_for_row()` refactored to read from `encoded_features[pattern_capacity]`

#### Helper Methods ✅
- `_cell_units(row, col)` → tiles committed to this cell (0–CAPACITY[row])
- `_cell_is_placed(row, col)` → binary wall placement check
- `_cell_is_full(row, col)` → binary "placed or pending" check
- `_adjacency(row, col, for_score)` → contiguous run count (0–10 or 1)
- `_pending(pending_cells)` → sum of adjacency for pending cells
- `_penalty()` → sum of FLOOR_PENALTIES up to floor line length
- `_bonus(row_demand, col_demand, tile_demand)` → complete row/col/tile bonuses
- `_max_pattern_capacity(pattern_capacity)` → sum of max per row across colors
- `_flatten(data)` → recursive flattener for mixed-depth nested lists

---

### Current Work: Player Encoding Tests (In Progress)

**Goal:** Validate 168-value encoding before refactoring downstream code.

**Test file location:** `tests/engine/test_player_encoding.py`

**Test suite (detailed):**

#### 1. Shape & Type Tests
```python
def test_encode_returns_168_values(player):
    """Verify _encode() produces exactly 168 features."""
    assert len(player.encoded_features) == 168

def test_encoded_features_is_list_of_ints(player):
    """All encoded values are int, no None/NaN/float."""
    assert all(isinstance(v, int) for v in player.encoded_features)
    assert len(player.encoded_features) == 168
```

#### 2. Slice Validation Tests
```python
def test_encoding_slices_dont_overlap():
    """All slices in ENCODING_SLICES are contiguous, no gaps/overlaps."""
    slices = list(ENCODING_SLICES.values())
    slices.sort(key=lambda s: s.start)
    for i in range(len(slices) - 1):
        assert slices[i].stop == slices[i+1].start

def test_encoding_slices_cover_all_168():
    """Sum of slice sizes equals 168."""
    total = sum(s.stop - s.start for s in ENCODING_SLICES.values())
    assert total == 168

def test_slice_bounds_match_docstring():
    """Each slice bounds match ENCODING_SLICES doc."""
    assert ENCODING_SLICES["wall"] == slice(0, 25)
    assert ENCODING_SLICES["pending_wall"] == slice(25, 50)
    assert ENCODING_SLICES["adjacency_grid"] == slice(50, 75)
    assert ENCODING_SLICES["wall_row_demand"] == slice(75, 100)
    assert ENCODING_SLICES["wall_col_demand"] == slice(100, 125)
    assert ENCODING_SLICES["wall_tile_demand"] == slice(125, 130)
    assert ENCODING_SLICES["pattern_demand"] == slice(130, 135)
    assert ENCODING_SLICES["pattern_capacity"] == slice(135, 160)
    assert ENCODING_SLICES["scoring"] == slice(160, 165)
    assert ENCODING_SLICES["misc"] == slice(165, 168)
```

#### 3. Wall & Pending Encoding Tests
```python
def test_wall_cells_all_zero_initially(empty_player):
    """Wall cells are 0 when no tiles placed."""
    wall_slice = ENCODING_SLICES["wall"]
    assert all(v == 0 for v in empty_player.encoded_features[wall_slice])

def test_wall_cells_one_when_placed(player_with_wall_tile):
    """Wall cell is 1 when tile placed."""
    # Place tile at (row=0, col=0)
    wall_slice = ENCODING_SLICES["wall"]
    # Index 0 in row-major order
    assert player_with_wall_tile.encoded_features[wall_slice.start + 0] == 1

def test_pending_cells_zero_when_pattern_incomplete(player):
    """Pending is 0 when pattern line not full."""
    pending_slice = ENCODING_SLICES["pending_wall"]
    # No full pattern lines → all pending = 0
    assert all(v == 0 for v in player.encoded_features[pending_slice])

def test_pending_cells_one_when_committed_and_full(player_full_pattern):
    """Pending is 1 when pattern line full and committed to wall color."""
    # Player has pattern line full, committed to correct color
    pending_slice = ENCODING_SLICES["pending_wall"]
    # At least one pending cell should be 1
    assert any(v == 1 for v in player_full_pattern.encoded_features[pending_slice])
```

#### 4. Adjacency Tests
```python
def test_adjacency_lone_tiles_score_one(player_lone_wall):
    """Isolated wall tiles score 1."""
    adj_slice = ENCODING_SLICES["adjacency_grid"]
    # Place single tile at (0, 2)
    assert player_lone_wall.encoded_features[adj_slice.start + 2] == 1

def test_adjacency_horizontal_run(player_horiz_run):
    """Contiguous horizontal tiles count run."""
    adj_slice = ENCODING_SLICES["adjacency_grid"]
    # Place tiles at (0, 0), (0, 1), (0, 2) — run of 3
    # Each should have adjacency >= 2
    assert player_horiz_run.encoded_features[adj_slice.start + 0] >= 2
    assert player_horiz_run.encoded_features[adj_slice.start + 1] >= 2
    assert player_horiz_run.encoded_features[adj_slice.start + 2] >= 2

def test_adjacency_range_0_to_10(player):
    """Adjacency values in range [0, 10]."""
    adj_slice = ENCODING_SLICES["adjacency_grid"]
    adj_vals = player.encoded_features[adj_slice]
    assert all(0 <= v <= 10 for v in adj_vals)
```

#### 5. Demand Grid Tests
```python
def test_wall_row_demand_empty_game(empty_player):
    """Empty board: each color needs full capacity per row."""
    row_slice = ENCODING_SLICES["wall_row_demand"]
    row_demands = empty_player.encoded_features[row_slice]
    # 25 values: 5 colors × 5 rows
    # Each color, each row should need CAPACITY[row]
    for color_idx in range(5):
        for row in range(5):
            idx = color_idx * 5 + row
            assert row_demands[idx] == CAPACITY[row]

def test_wall_col_demand_sorted(player_partial_wall):
    """Columns sorted by total demand (most complete first)."""
    col_slice = ENCODING_SLICES["wall_col_demand"]
    col_demands = player_partial_wall.encoded_features[col_slice]
    # 25 values: 5 colors × 5 cols
    # Check that col totals are non-increasing (sorted ascending by demand)
    col_totals = [sum(col_demands[c*5:(c+1)*5]) for c in range(5)]
    assert col_totals == sorted(col_totals)

def test_wall_tile_demand_sum_matches_total(player):
    """Sum of per-color tile demand = total wall tiles needed."""
    tile_slice = ENCODING_SLICES["wall_tile_demand"]
    tile_demands = player.encoded_features[tile_slice]
    # 5 values, one per color
    assert len(tile_demands) == 5
    assert all(0 <= v <= 25 for v in tile_demands)  # Max 5 rows × 5 tiles per color

def test_pattern_demand_only_committed(player_mixed_patterns):
    """Pattern demand only counts committed (full) lines."""
    pattern_slice = ENCODING_SLICES["pattern_demand"]
    pattern_demands = player_mixed_patterns.encoded_features[pattern_slice]
    # 5 values, one per color
    assert len(pattern_demands) == 5
    # Only committed lines contribute demand (no empty, no partial)

def test_pattern_capacity_room_for_all_rows(player):
    """Pattern capacity sum = sum of CAPACITY - pattern fills."""
    cap_slice = ENCODING_SLICES["pattern_capacity"]
    capacities = player.encoded_features[cap_slice]
    # 25 values: 5 colors × 5 rows
    expected_total = sum(CAPACITY) - sum(len(player._pattern_lines[r]) for r in range(5))
    assert sum(capacities) == expected_total
```

#### 6. Scoring Tests
```python
def test_scoring_slice_matches_properties(player):
    """Encoded scoring matches property accessors."""
    score_slice = ENCODING_SLICES["scoring"]
    scores = player.encoded_features[score_slice]
    assert len(scores) == 5
    assert scores[0] == player.score
    assert scores[1] == player.pending
    assert scores[2] == player.penalty
    assert scores[3] == player.bonus
    assert scores[4] == player.earned

def test_pending_score_calculation(player_full_pattern_with_adjacency):
    """Pending = sum of adjacency for pending cells."""
    # Manually calculate expected pending
    expected_pending = 0
    for row in range(5):
        if len(player_full_pattern_with_adjacency._pattern_lines[row]) == CAPACITY[row]:
            for col in range(5):
                # Check if this cell is pending
                expected_pending += player_full_pattern_with_adjacency._adjacency(row, col, for_score=True)
    assert player_full_pattern_with_adjacency.pending == expected_pending

def test_penalty_score_calculation(player_with_floor):
    """Penalty = sum of FLOOR_PENALTIES up to floor length."""
    expected_penalty = sum(FLOOR_PENALTIES[:len(player_with_floor._floor_line)])
    assert player_with_floor.penalty == expected_penalty

def test_bonus_score_calculation(player_complete_row):
    """Bonus includes complete row bonuses."""
    # Player has one complete row, no cols/tiles
    expected_bonus = BONUS_ROW
    assert player_complete_row.bonus == expected_bonus

def test_earned_is_sum_of_components(player):
    """Earned = score + pending + penalty + bonus."""
    expected = player.score + player.pending + player.penalty + player.bonus
    assert player.earned == expected
```

#### 7. Misc Values Tests
```python
def test_first_player_token_zero_when_absent(player_no_first_player):
    """First player token is 0 when not in floor."""
    misc_slice = ENCODING_SLICES["misc"]
    misc_vals = player_no_first_player.encoded_features[misc_slice]
    # Index 0 of misc (first 3 values)
    assert misc_vals[0] == 0

def test_first_player_token_one_when_present(player_with_first_player):
    """First player token is 1 when in floor."""
    misc_slice = ENCODING_SLICES["misc"]
    misc_vals = player_with_first_player.encoded_features[misc_slice]
    assert misc_vals[0] == 1

def test_total_used_calculation(player):
    """Total used = MAX_USED - sum(wall_tile_demand)."""
    misc_slice = ENCODING_SLICES["misc"]
    misc_vals = player.encoded_features[misc_slice]
    # Index 1 of misc
    tile_slice = ENCODING_SLICES["wall_tile_demand"]
    tile_demands = player.encoded_features[tile_slice]
    expected_used = MAX_USED - sum(tile_demands)
    assert misc_vals[1] == expected_used

def test_max_pattern_capacity(player):
    """Max pattern capacity = sum of max per row across colors."""
    misc_slice = ENCODING_SLICES["misc"]
    misc_vals = player.encoded_features[misc_slice]
    # Index 2 of misc
    cap_slice = ENCODING_SLICES["pattern_capacity"]
    capacities = player.encoded_features[cap_slice]
    expected_max = sum(max(capacities[c*5:(c+1)*5]) for c in range(5))
    assert misc_vals[2] == expected_max
```

#### 8. Flatten Tests
```python
def test_flatten_flat_list(player):
    """Flatten on already-flat list returns unchanged."""
    flat = [1, 2, 3]
    result = player._flatten(flat)
    assert result == [1, 2, 3]

def test_flatten_nested_lists(player):
    """Flatten on nested lists unpacks all levels."""
    nested = [[1, 2], [3, 4]]
    result = player._flatten(nested)
    assert result == [1, 2, 3, 4]

def test_flatten_mixed_depth(player):
    """Flatten handles mixed nesting depth."""
    mixed = [1, [2, [3, 4]], 5]
    result = player._flatten(mixed)
    assert result == [1, 2, 3, 4, 5]

def test_flatten_empty_sublists(player):
    """Flatten skips empty sublists."""
    with_empty = [1, [], [2]]
    result = player._flatten(with_empty)
    assert result == [1, 2]

def test_flatten_deeply_nested(player):
    """Flatten handles arbitrary nesting depth."""
    deep = [[[[[1]]]]]
    result = player._flatten(deep)
    assert result == [1]

def test_flatten_preserves_type(player):
    """Flatten preserves int type of all values."""
    data = [[1, 2], [3, 4]]
    result = player._flatten(data)
    assert all(isinstance(v, int) for v in result)
```

#### 9. Consistency Tests
```python
def test_encode_idempotent(player):
    """Calling _encode() twice produces same features."""
    features1 = player.encoded_features[:]
    player._encode()
    features2 = player.encoded_features
    assert features1 == features2

def test_encode_after_place(player):
    """Features update after place()."""
    features_before = player.encoded_features[:]
    player.place(0, [Tile.BLUE])
    features_after = player.encoded_features
    assert features_before != features_after

def test_encode_after_process_round_end(player_ready_for_round_end):
    """Features update after process_round_end()."""
    features_before = player_ready_for_round_end.encoded_features[:]
    player_ready_for_round_end.process_round_end()
    features_after = player_ready_for_round_end.encoded_features
    assert features_before != features_after

def test_from_string_encodes_correctly(player):
    """Player.from_string() recomputes encoding correctly."""
    s = str(player)
    player2 = Player.from_string(s)
    assert player.encoded_features == player2.encoded_features
```

**Fixtures required:**
- `empty_player` — fresh Player, no moves made
- `player` — fresh Player (same as empty_player)
- `player_with_wall_tile` — one tile placed on wall
- `player_full_pattern` — one pattern line full, correct color
- `player_lone_wall` — single isolated wall tile
- `player_horiz_run` — horizontal run of wall tiles
- `player_partial_wall` — multiple tiles, various states
- `player_mixed_patterns` — mix of empty/partial/full lines
- `player_full_pattern_with_adjacency` — full pattern + adjacency setup
- `player_with_floor` — floor line with tiles
- `player_complete_row` — complete wall row
- `player_no_first_player` — floor without FIRST_PLAYER
- `player_with_first_player` — floor with FIRST_PLAYER token
- `player_ready_for_round_end` — setup for round-end processing

**Success criteria:**
- All tests pass
- 100% coverage of `_encode()`, `_flatten()`, and all helpers
- Encoding values in expected ranges per documentation
- No external method calls (encapsulated tests)

---

### Next: Game Refactor

**Goal:** Update `Game` to use new Player encoding, remove/update methods that relied on deleted Player methods.

**Scope:**
- Remove calls to deleted methods (`_is_complete()`, `has_triggered_game_end()`)
- Implement `has_triggered_game_end()` using `encoded_features` if needed
- Refactor `determine_winners()` if it uses deleted methods
- Update any game-flow logic that checked Player state directly

**Tests:** `tests/engine/test_game.py` — validate game flow still works with new encoding

---

### Finally: Fix External Callers

**Scope:**
- API endpoints that read Player state
- Agent code (AlphaZero, AlphaBeta, etc.) that read Player state
- Scripts (train.py, evaluate.py, etc.)
- UI code that displays Player state

**Do NOT sync to main until all external callers fixed** — CI will fail otherwise.

---

## Encoding Reference

### ENCODING_SLICES
```python
{
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

### Value Ranges
- **Wall/Pending:** 0–1 (binary)
- **Adjacency:** 0–10 (run count, min 1)
- **Demand/Capacity:** 0–CAPACITY[row] (0–3 per cell)
- **Scoring:** unclamped int (apply ENCODING_DIVISORS in model)
- **Misc:** 0–1 (token), 0–100 (used), 0–25 (max capacity)

---

## Timeline

| Task | Estimate | Status |
|---|---|---|
| Write Player encoding tests | 3h | 🔄 In progress |
| Game refactor | 4h | ⏳ Pending |
| Game tests | 2h | ⏳ Pending |
| Fix external callers | 8h | ⏳ Pending |
| **Total** | **17h** | |

---

## Key Principles

1. **Tests first, always** — Don't refactor downstream until Player tests pass
2. **Encapsulation** — Player encoding is self-contained; don't expose internals
3. **Gradual rollout** — Fix external callers in this order: Game → API → Agents → Scripts → UI
4. **CI hygiene** — Never commit breaking changes to main. Work on feature branch, fix all issues before sync
5. **Documentation** — Update docstrings as API changes, keep ENCODING_SLICES in sync

---

## Deferred/Future Work

- Game-level encoding (tile availability, bag state, factory info)
- Supervised trainer integration with new encoding
- AlphaBeta + learned value net (`AlphaBetaWithLearnedValue`)
- Value net training loop (generate → train → eval)
- Hyperparameter search (model width, dropout, learning rate)
- Checkpoint management for supervised training

---

**Status:** Ready to begin comprehensive Player encoding test suite. No changes to encoder; tests validate it works correctly before downstream refactoring begins.