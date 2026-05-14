# Azul: Supervised Learning Value Net — Project Plan

**Last Updated:** 2026-05-12  
**Branch:** `feat/supervised-value-net`  
**Status:** Phase 0 in progress — Player encoding implementation 101+ values complete. Refactored to use INDICES_BY for bonus computation.

---

## Executive Summary

This project pivots the AlphaZero training pipeline from self-play reinforcement learning to **supervised learning with AlphaBeta agents**. Instead of millions of games, we generate training data by:

1. Playing full games with AlphaBeta agents (with low-temperature stochasticity)
2. Collecting every played position + labeling with final game outcome
3. Training a value-only neural net (`AzulValueNet`) on (state, value) pairs
4. Using that net inside AlphaBeta (`AlphaBetaWithLearnedValue`) as the eval function
5. Repeating for up to 10 generations, stopping when win rate plateaus

**Why this works for Azul:** Azul is shallow (~30 moves, ~6 rounds). We can bootstrap from hand-coded AB heuristics and improve iteratively.

---

## Data Generation Strategy

### Approach: Deterministic AB with Low Temperature

Generate training data by playing complete games with AlphaBeta agents:

1. **Play games:** Two `AlphaBetaAgent(depth=3, threshold=6)` agents play each other to completion
2. **Collect every played position:** For each move in the game, collect the pre-move state
3. **Label with final outcome:** Label all positions with the final score differential (from current player's perspective)
4. **Stochastic move selection:** AB uses softmax temperature `temp=0.1-0.3` over its evaluated moves
   - ~95% of moves: AB's best-evaluated move
   - ~5% of moves: occasionally picks a slightly-suboptimal move
   - Provides natural exploration without artificial noise
   - If a deviant move leads to good outcome, future generations will learn to favor it

**Data per generation:**
- 50 games → ~1500 examples (30 moves × 50 games)
- Cost: ~30 minutes
- Label quality: All examples labeled with true game outcome (noisy but signal-rich)

**Why this approach:**
- Simple to implement (one parameter change to AB)
- Data remains "pure" (no artificial forks or rollouts)
- Explores naturally (occasionally tries non-greedy moves)
- Supports bootstrapping (Gen-1 can learn to prefer deviations if they're actually good)

---

## Encoding Spec — Locked

### Overview

**Single player encodes 101+ values about their own board state.**

The full game encoding will be:
- `[current_player.encode() (101+)] + [opponent.encode() (101+)] + [game.encode() (TBD)]`

**Total per-player: 101+ values** (final count depends on remaining sections)

### Player Encoding Structure

Encoding is always from the **current player's perspective**. Each section is computed once during `_encode()` and stored in `encoded_features` list.

#### Section 1: Wall (25 values, indices 0–24)
Binary 0/1 for each wall cell indicating if a tile is placed there.
Layout: row-major order (0–4 = row 0, 5–9 = row 1, etc.).
```python
[1 if self._wall_tiles[row][col] is not None else 0 for row in range(SIZE) for col in range(SIZE)]
```

#### Section 2: Pattern Fill Units (25 values, indices 25–49)
Raw tile count (0 to CAPACITY[row]) for each pattern line.
For each wall cell, if the pattern line for that row is aimed at that cell's tile color, the count; else 0.
Layout: row-major order (matches wall layout).
```python
[
    len(self._pattern_lines[row])
    if (self._pattern_lines[row] and self._pattern_lines[row][0] == TILE_FOR_ROW_COL[row][col])
    else 0
    for row in range(SIZE) for col in range(SIZE)
]
```

#### Section 3: Pending Wall (25 values, indices 50–74)
Binary 1/0 for each wall cell indicating if that cell will be placed at round end.
Requires: pattern line at full capacity for that cell's tile.
Layout: row-major order.
```python
[1 if pattern_fill_counts[row * SIZE + col] == CAPACITY[row] else 0 for row in range(SIZE) for col in range(SIZE)]
```

#### Section 4: Adjacency Grid (25 values, indices 75–99)
Placement score for each wall cell: sum of contiguous runs (horizontal + vertical).
Lone tiles score 1. Range 0–10.
Computed using encoded wall and pending wall arrays; no need to re-check state.
Layout: row-major order.

#### Section 5: Bonus Completion Units (15 values, indices 100–114)
For each feature group (5 rows + 5 cols + 5 tiles), completion progress.
Each cell contributes either CAPACITY[row] (if placed on wall) or its pattern fill count.
Range 0–5 per feature (5 means the feature will score a bonus).
```python
[
    sum(
        CAPACITY[i // SIZE] if wall_encoded[i] else pattern_fill_counts[i]
        for i in feature_indices
    )
    for feature_type in ['row', 'col', 'tile']
    for feature_indices in INDICES_BY[feature_type]
]
```

**Note:** Order is rows (0–4), then cols (0–4), then tiles (0–4).

#### Section 6: Pattern Completion Flags (5 values, indices 115–119)
Binary 1/0 per pattern line indicating if it is full.
```python
[1 if len(self._pattern_lines[row]) == CAPACITY[row] else 0 for row in range(SIZE)]
```

#### Section 7: Scoring (5 values, indices 120–124)
- **Index 120:** score (confirmed points from prior rounds)
- **Index 121:** pending_score (sum of adjacency points for cells being placed this round)
- **Index 122:** penalty (negative floor line length)
- **Index 123:** bonus_score (points earned from completing rows/cols/tiles this round)
- **Index 124:** earned (score + pending_score + penalty + bonus_score)

#### Section 8: Tiles Needed (1 value, index 125)
Sum across all started (non-empty, incomplete) pattern lines of tiles still needed.
```python
sum(
    CAPACITY[row] - len(self._pattern_lines[row])
    for row in range(SIZE)
    if self._pattern_lines[row]
)
```

### Remaining Sections (TODO)

- **First Player Token** (1): 1 if holding first-player marker, else 0
- **Wall Completion Progress** (15): fraction of each row/col/tile that is complete (by cells)
- **Top Completions** (6): sorted descending completion fractions (top 3 rows, top 2 cols, top 1 tile)
- **Incomplete Lines Count** (1): count of pattern lines that have tiles but are not full
- **Pattern Line Demand** (5): per color, how many tiles needed to complete all started lines for that color
- **Wall Completion Demand** (30): for each top-completion group, per-color demand (6 groups × 5 colors)
- **Adjacency Demand** (5): per color, sum of adjacency scores for all empty wall cells of that color
- **Total Used Tiles** (1): fraction of the 100-tile pool already committed to wall + pattern lines

---

## Implementation Notes

### INDICES_BY Structure

A unified dict mapping feature types to wall cell indices:
```python
INDICES_BY = {
    'row': [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], ...],  # 5 rows
    'col': [[0, 5, 10, 15, 20], [1, 6, 11, 16, 21], ...],  # 5 cols
    'tile': [[cells for tile 0], [cells for tile 1], ...],  # 5 tiles
}
```

Built once at module load from `CELLS_BY_TILE` constant. Enables efficient bonus computation without re-checking player state.

### next_rounds_wall

Computed in `_encode()` as `[wall_encoded[i] + pending_encoded[i] for i in range(SIZE * SIZE)]`.
Represents the final wall state after this round's placements.
Passed to bonus computation methods to avoid redundant array building.

### Encoding Flow

1. Compute `wall_encoded` (section 1)
2. Compute `pattern_fill_counts` (section 2)
3. Compute `pending_encoded` from `pattern_fill_counts` (section 3)
4. Compute `adjacency_encoded` using `wall_encoded` and `pending_encoded` (section 4)
5. Compute `next_rounds_wall` as sum of wall and pending
6. Compute `bonus_completion_units` from `wall_encoded` and `pattern_fill_counts` (section 5)
7. Compute `pattern_completion_flags` directly (section 6)
8. Compute scoring values: pending_score, bonus_score, then score, penalty, earned (section 7)
9. Compute `tiles_needed` directly (section 8)
10. TODO: Compute remaining sections

Each section is extended directly into `features` list. No normalization happens here; that's the encoder/model layer's job.

---

## Development Phases

### Phase 0 — Player Encoding Implementation 🔄 (in progress)

**Goal:** Complete implementation of `Player.encode()` method with all feature sections.

**Completed:**
- [x] Player refactored: `_pattern_lines: list[list[Tile]]`, `_wall_tiles: list[list[Tile | None]]`
- [x] `encoded_features` field storing raw int values
- [x] Fixed-width 23-char display (`__str__`) and round-trip reconstruction (`from_string()`)
- [x] Sections 1–8 implemented (101+ values)
- [x] INDICES_BY structure created and tested
- [x] Bonus computation refactored to use INDICES_BY

**Remaining:**
- [ ] Implement sections 9–17 (remaining 50+ values)
- [ ] Update scoring properties to read from `encoded_features` slices
- [ ] Write comprehensive tests for all encode sections
- [ ] Verify round-trip: game → encode → values match expected ranges

**Next step:**
- Implement remaining encoding sections (First Player Token through Total Used Tiles)

---

### Phase 1 — Data Generation Pipeline

**Goal:** Generate labeled (state, value) training examples using greedy AB play with low-temperature exploration.

**Tasks:**
1. Modify `AlphaBetaAgent.choose_move()` to support temperature parameter
2. Implement `scripts/generate_supervision_data.py`
3. Smoke test: verify ~30 min runtime for 50 games
4. Validate: all examples have valid shape, value distribution roughly Gaussian

---

### Phase 2 — Value Network and Training

**Goal:** `AzulValueNet` + `SupervisedValueTrainer` with architecture comparison

**Architecture options:**

| Architecture | Layers | Params |
|---|---|---|
| Wide | 101+ → 64 → 1 | ~7k–10k |
| Narrow | 101+ → 32 → 8 → 1 | ~4k–5k |

Both use: ReLU activations, dropout 0.1, sigmoid output, MSE loss, Adam optimizer, early stopping (patience=5).

---

### Phase 3 — Agent Integration

**Goal:** `AlphaBetaWithLearnedValue` — AB search using the value net as its eval function instead of hand-coded `earned`.

---

### Phase 4 — Evaluation Framework

**Goal:** Robust eval harness: 100 mirrored pairs, win rate + 95% CI, results to JSON.

---

### Phase 5 — Generational Loop

**Goal:** Automated pipeline with plateau detection. Runs overnight unattended.

**Pipeline per generation:**
1. Generate 1500 examples (Phase 1)
2. Train value net (Phase 2)
3. Integrate into AB agent (Phase 3)
4. Evaluate 100 mirrored pairs (Phase 4)
5. Decide: continue or stop

**Plateau detection:**
- Stop if <2% win rate improvement for 3 consecutive generations
- Hard cap: 10 generations

---

### Phase 6 — UI Integration

**Goal:** Inspector dropdown to select learned value net generation. Game UI playable against `AlphaBetaWithLearnedValue`.

---

## Success Criteria

| Milestone | Metric | Target | Status |
|---|---|---|---|
| **Phase 0** | Encoding implemented and tested | 100% | 🔄 In Progress |
| **Phase 1** | Data generated | 1500+ examples | ⏳ Not Started |
| **Phase 2** | Gen-0 training | Val r > 0.85, MSE < 0.15 | ⏳ Not Started |
| **Phase 2** | Architecture comparison | Winner selected | ⏳ Not Started |
| **Phase 3** | Agent integration | Compiles and plays | ⏳ Not Started |
| **Phase 4** | Gen-0 eval | Win rate ≥ 55% | ⏳ Not Started |
| **Phase 5** | Gen-1 eval | Win rate ≥ 60% | ⏳ Not Started |
| **Phase 5** | Gen-2 eval | Win rate ≥ 65% | ⏳ Not Started |
| **Phase 5** | Gen-3+ eval | Win rate ≥ 75% | ⏳ Not Started |
| **Final** | Win rate vs AB hard | ≥ 85% (aspirational) | ⏳ Not Started |
| **Final** | Pipeline | All tests passing | ⏳ Not Started |

---

## Hard-Won Lessons (carry forward from AlphaZero pipeline)

1. **`earned` is the dominant signal** — AB uses only `earned` and beats all other heuristic agents. Include it explicitly; don't make the net derive it.
2. **Per-instance RNG everywhere** — `random.Random()` per agent and per game. Never global `random.seed()`. Global state causes mirror pairs to produce identical scores.
3. **Windows: no Unicode in log strings** — use ASCII only in logging calls.
4. **`uvicorn --reload` restarts on checkpoint writes** — run without `--reload` during training.
5. **`Move` uses `.tile` not `.color`** — always.
6. **Import `Tile` from `engine.constants`** — never from `engine.tile`.
7. **Game RNG must be instance-local** — affects which tiles are drawn and factories filled. Global state ruins determinism.
8. **Perspective matters** — value is always from current player's perspective at that position. Flip when needed.

---

## Open Questions

- Should `AlphaBetaWithLearnedValue` use the net at every node, or only at leaf nodes?
  - Leaf-only is faster; every node is more accurate but slower.
- Should we keep the existing AlphaZero pipeline intact on `main` and merge only after supervised learning proves competitive?
- After 10 generations, keep all checkpoints or archive?
- If plateau detection stops early (Gen-3), investigate: is the problem the encoding, the architecture, or the label noise?
- Final count of encoding features: 101+ (remaining sections TBD)

---

## Repository Structure (Updated)

```
azul-alphazero/
├── engine/
│   ├── player.py                        # Refactored: _encode() with 101+ features
│   ├── constants.py                     # Updated: INDICES_BY dict
│   └── game.py                          # Will add game.encode()
│
├── neural/
│   ├── model.py                         # AzulValueNet (value-only)
│   └── trainer.py                       # SupervisedValueTrainer
│
├── agents/
│   └── alphabeta_learned.py             # AlphaBetaWithLearnedValue
│
├── scripts/
│   ├── generate_supervision_data.py     # NEW
│   ├── train_value_net.py               # NEW
│   ├── evaluate_agents.py               # NEW
│   └── train_generations.py             # NEW (orchestrator)
│
└── checkpoints/supervised/
    ├── gen_0000/
    │   ├── best_checkpoint.pt
    │   ├── training_log.csv
    │   └── eval_results.json
    └── latest.pt
```

---

## Timeline Estimate

| Phase | Wall Time | Dependencies |
|---|---|---|
| **0** | ~4 hours | Encoding implementation + tests |
| **1** | ~30 min + overnight | Phase 0 complete |
| **2** | ~1 hour | Phase 1 complete |
| **3** | ~2 hours | Phase 2 complete |
| **4** | ~2 hours | Phase 3 complete |
| **5** | ~2–7 days (unattended) | Phase 4 complete |
| **6** | ~2 hours | Phase 5 complete |
| **Total manual work** | ~11 hours + ~5 days overnight | |

---

**Document Status:** Phase 0 encoding implementation ongoing. Core structure locked, remaining sections identified. Ready to implement sections 9–17 after current batch completes.