# Phase 0: Supervised Learning Refactor

**Last Updated:** 2026-05-15  
**Status:** Player encoding tests in progress  
**Branch:** `feat/supervised-value-net`

---

## Objective

Validate and refactor the player encoding from the AlphaZero era to support the new supervised learning approach. The 168-value encoding [[encoding_strategy.md]] has been implemented but downstream code hasn't been updated yet.

**Key constraint:** Don't touch downstream until Player tests pass. Refactoring out of order breaks CI.

---

## Phases of Phase 0

### Phase 0a — Player Encoding Tests ✅ (Current)

**Status:** Tests implemented, running  
**Commits:** Recent test suite in `tests/engine/test_player.py`

**What we're validating:**
- Encoding shape (exactly 168 values)
- All values are integers (no None, NaN, float)
- Slices in ENCODING_SLICES are contiguous, no gaps
- Wall/pending cells match actual board state
- Adjacency counts are correct (0–10 range)
- Demand grids reflect actual tiles needed
- Scoring values match properties (earned = score + pending + penalty + bonus)
- Misc values (first player, used tiles, max capacity)
- Flatten helper handles nested lists
- Encoding is idempotent (same board → same encoding)
- Encoding updates after `place()` and `process_round_end()`

**Test file:** `tests/engine/test_player.py`

**Exit criteria:**
- All tests pass
- 100% coverage of `Player._encode()` and helpers
- Encoding contract is solid before touching Game

**Estimate:** Complete this week

---

### Phase 0b — Game Refactor ⏳ (Next)

**Status:** Pending Phase 0a completion  
**Scope:** Update `engine/game.py` for new encoding

**Changes needed:**
1. Remove calls to deleted Player methods (if any)
2. Verify `determine_winners()` works with new encoding
3. Update any game-flow checks that read Player fields directly
4. Ensure `advance()` and `advance(skip_setup=True)` still work

**Test strategy:** Run `tests/engine/test_game.py` to verify game flow

**Estimate:** 4 hours

---

### Phase 0c — Game Tests ⏳ (Next)

**Status:** Pending Phase 0b

**Scope:** Comprehensive game tests with new encoding

**Coverage:**
- Game initialization and round setup
- Move legality and validation
- Round transitions and scoring
- Game end detection and winner determination
- Mirror pairs (identical seed, sides swapped)

**Test file:** `tests/engine/test_game.py`

**Estimate:** 2 hours

---

### Phase 0d — External Caller Fixes ⏳ (Final)

**Status:** Pending Phases 0a–0c

**Scope:** Fix all code outside engine/ that reads Player state

**Files to update (in order):**

#### 1. API Endpoints (`api/main.py`)
- Responses that read `player.wall`, `player.pattern`, `player.score`
- Game state serialization
- Estimate: 2 hours

#### 2. Agents (`agents/`)
- AlphaBeta value function reads `player.earned`
- MCTS leaf evaluation
- Any heuristic agents reading board state
- Estimate: 2 hours

#### 3. Scripts (`scripts/`)
- `train.py` — game collection, buffer management
- `log_encoded_states.py` — state visualization
- `inspect_policy.py` — policy inspection
- Estimate: 2 hours

#### 4. UI / Frontend (`api/main.py` responses)
- Game state for display
- Ensure all fields exist and serialize correctly
- Estimate: 2 hours

**Exit criteria:**
- All tests pass
- No import errors
- CI green

**Total estimate:** ~17 hours across all phases

---

## Critical Gotchas

### Don't Break Encapsulation
Player fields are now private (`_wall`, `_pattern`, etc.) with property accessors. Code reading `player._wall` directly will break. Use public properties instead.

### Read Earned Before Advance
```python
# WRONG (after advance)
value = player.earned  # bonus already folded into score, double-counts

# CORRECT (before advance)
earned_before = player.earned
game.advance(skip_setup=True)
```

See [[alphabeta_strategy.md#round_boundaries]] for details.

### Round Boundaries for Tree Search
AlphaBeta uses `advance(skip_setup=True)` to stay at round boundaries. Never call `advance()` without arguments inside tree search — factories will be refilled and position becomes meaningless.

### Encoding Updates Automatically
Once a field is changed (via `place()` or round-end processing), `encoded_features` is automatically recomputed. You don't need to manually call `_encode()` — it happens on mutation.

---

## Testing Approach

### Test Files Organization
```
tests/
├── engine/
│   ├── test_game.py          # Game flow, round transitions, scoring
│   └── test_player.py        # Player encoding, properties, board state
├── agents/
│   ├── test_alphabeta.py     # AlphaBeta move selection, value eval
│   └── test_minimax.py       # Minimax tree search
└── test_api.py               # API responses and state serialization
```

### Running Tests
```bash
# All tests
pytest

# Specific file
pytest tests/engine/test_player.py

# Specific test
pytest tests/engine/test_player.py::TestScoringProperties::test_earned_is_sum_of_components

# With coverage
pytest --cov=engine --cov=agents
```

### Success Criteria
- All tests pass
- No deprecation warnings
- Coverage > 95% for modified code
- CI green on branch and main

---

## Dependency Graph

```
Phase 0a (Player tests)
    ↓
Phase 0b (Game refactor)
    ↓
Phase 0c (Game tests)
    ↓
Phase 0d (External callers)
    ↓
Merge to main
    ↓
Phase 1 (Supervised Training)
```

**Cannot skip phases.** Each phase depends on the previous one passing.

---

## Rollout Strategy

### During Development
- Work on feature branch `feat/supervised-value-net`
- Push regularly to GitHub (runs CI on PR)
- If CI fails, diagnose and fix before next phase

### When Ready to Merge
- All tests pass locally and on CI
- No regressions in other branches
- Code review (if applicable)
- Merge to main

### If Blockers Arise
- Document the blocker in a comment on the branch
- Create a separate task if needed
- Don't force a fix that hides the real problem

---

## Encoding Integration Points

### Where Encoding Is Read
1. **Supervised trainer** (Phase 1) — Input to neural net
2. **AlphaBeta value function** (Phase 1) — Replaces earned-score heuristic
3. **Inspector UI** (future) — Display encoded features
4. **Tests** — Validate correctness

### Where Encoding Is Written
1. **Player.place()** — Places tile on pattern/floor
2. **Player.process_round_end()** — Pattern→wall transfer, scoring

### Encoding Contract
```python
@property
def encoded_features(self) -> list[int]:
    """Return 168-value encoding of board state. Always up-to-date."""
    return self._cached_features
```

The contract: read `encoded_features` at any time and get the current state. No manual refresh needed.

---

## Deferred Work

**These are out of scope for Phase 0:**
- Game-level encoding (tile availability, bag state, factory info)
- Supervised trainer integration with new encoding
- AlphaBeta + learned value net
- Value net training loop
- Hyperparameter search

These become Phase 1 work once Phase 0 is complete.

---

## References

- [[master_plan.md]] — High-level vision and phases
- [[encoding_strategy.md]] — 168-value encoding specification
- [[alphabeta_strategy.md]] — AlphaBeta design and round boundaries

---

## Checklist

### Pre-Phase 0a (Setup)
- [x] Encoding implemented in `engine/player.py`
- [x] `ENCODING_SLICES` defined
- [x] Tests written

### Phase 0a (Current)
- [ ] All Player tests pass
- [ ] 100% coverage of encoding
- [ ] No breaking changes to Player public API
- [ ] Branch CI green

### Phase 0b (Game Refactor)
- [ ] Game.py updated for new encoding
- [ ] No Player method calls deleted
- [ ] All game tests pass

### Phase 0c (Game Tests)
- [ ] Comprehensive game flow tests
- [ ] Round transitions verified
- [ ] Winner determination correct

### Phase 0d (External Callers)
- [ ] API endpoints updated
- [ ] Agent code updated
- [ ] Scripts updated
- [ ] UI state serialization correct
- [ ] All tests pass
- [ ] CI green
- [ ] Ready to merge to main

---

**Status:** Ready to begin comprehensive Player encoding test suite. Awaiting completion before proceeding to Phase 0b.
