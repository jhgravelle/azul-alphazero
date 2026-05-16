# Test Best Practices for Azul AlphaZero

**Lessons learned from fixing 47 test failures** during engine refactoring integration (2026-05-15).

---

## Overview

When the Player class was refactored from `_pattern_grid` → `_pattern_lines` and `_wall` → `_wall_tiles`, many tests broke not because of the refactoring itself, but because tests were **accessing private implementation details** and **manually setting invalid game states**.

This document captures the patterns that prevented failures and the anti-patterns that caused them.

---

## Anti-Patterns to Avoid

### 1. Manually Setting Private Fields

**❌ NEVER do this:**

```python
# Direct assignment to private attributes
player._wall_tiles[2][0] = Tile.BLUE
player._pattern_lines[2] = [Tile.RED, Tile.RED]
player._floor_line = [Tile.YELLOW, Tile.YELLOW]
player._encode()
```

**Why it fails:**
- Creates **invalid game states** that violate rules
- Example: You set wall[2] = BLUE, then try to place RED in row 2 via `make_move()` → fails because BLUE is already there
- Fragile to refactoring: if `_pattern_lines` changes to a different structure, test breaks immediately

**✅ Instead, use public APIs:**

```python
# Use place() for pattern lines
player.place(0, [Tile.WHITE])
player.place(1, [Tile.BLUE, Tile.BLUE])

# Use process_round_end() to finalize and score
player.process_round_end()

# Or use Game.from_string() for complex scenarios
game = Game.from_string("""...""")
```

---

### 2. Accessing Stale Cached Values

**❌ NEVER do this:**

```python
# Manual field change without _encode()
player.score = 40
assert player.earned > 30  # FAILS — earned is stale!
```

**Why it fails:**
- `earned` is computed from `encoded_features`, which only updates when `_encode()` is called
- Manual field changes don't trigger `_encode()` automatically
- Test fails with confusing assertion error

**✅ Instead, call _encode() after changes:**

```python
# Manual field change must be followed by _encode()
player.score = 40
player._encode()
assert player.earned > 30  # PASSES — cache is fresh
```

---

### 3. Depending on Removed Private Methods

**❌ NEVER do this:**

```python
# These methods no longer exist
player._update_penalty()
player._update_score()
player._update_bonus()
```

**Why it fails:**
- Methods were removed in refactoring
- Test fails immediately with `AttributeError`

**✅ Instead, use _encode():**

```python
# Single method recomputes all metrics
player._encode()
```

---

### 4. Creating Impossible Game States

**❌ NEVER do this:**

```python
# Build wall with BLUE in row 2
_place_wall(player._wall_tiles, 2, Tile.BLUE)

# Then try to place BLUE in row 2
game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=2))
# AssertionError: BLUE already on wall in row 2!
```

**Why it fails:**
- Violates game rules: each row can only have one of each color
- Manual wall setup creates states that can't legally occur
- Test setup is more complex than the game flow it tries to test

**✅ Instead, use place() to build state through legal moves:**

```python
# Build state naturally
game = Game()
game.setup_round()
game.players[0].place(0, [Tile.WHITE])
game.players[0].place(1, [Tile.BLUE, Tile.BLUE])
game.players[0].place(2, [Tile.RED, Tile.RED, Tile.RED])
# All states are guaranteed legal
```

---

## Patterns That Work

### 1. Use Game.from_string() for Complex Positions

**✅ DO this for known scenarios:**

```python
game = Game.from_string("""
Round 1
Factories: [W W B B] [R R Y Y] [K K W W] [...]
Center: [B]
Players:
  0: Scores 27, Wall rows 0-2 complete, Pattern lines 3-4 full
  1: Scores 23, ...
""")
```

**Benefits:**
- Readable and declarative
- State is guaranteed valid (parser ensures legality)
- Easy to adjust for different test scenarios
- Self-documenting: clear what state you're testing

---

### 2. Access Player State via Public Properties

**✅ DO this when checking state:**

```python
# Use public properties instead of private fields
assert player.wall[0][0] == Tile.BLUE
assert len(player.pattern_lines[1]) == 2
assert len(player.floor_line) == 3

# Or better: test behavior, not state
assert player.is_tile_valid_for_row(Tile.BLUE, 0) == False
```

**Benefits:**
- Safe from refactoring (properties hide implementation)
- Tests the contract, not the implementation
- Clearer test intent

---

### 3. Call _encode() When Manually Changing State

**✅ DO this if you must manually change state:**

```python
# Set a field
player.score = 40

# Immediately call _encode() to refresh cache
player._encode()

# Now derived values are up-to-date
assert player.earned > 30
```

**When this is acceptable:**
- Isolated test needing specific scoring scenario
- No way to create the state via public API
- Rare and clearly documented

---

### 4. Use place() and process_round_end() for Setup

**✅ DO this for natural state progression:**

```python
# Build position step-by-step
game = Game()
game.setup_round()

# P1's moves
game.players[0].place(0, [Tile.WHITE])
assert game.players[0].pattern_lines[0] == [Tile.WHITE]

# P2's moves
game.players[1].place(1, [Tile.BLUE, Tile.BLUE])

# End round
game.players[0].process_round_end()
game.players[1].process_round_end()

# All state changes are automatic
assert game.players[0].earned > 0  # Earned value computed
```

**Benefits:**
- State is always legal
- Automatic `_encode()` calls
- Test reads like real game flow

---

## Quick Checklist

Before submitting a test:

- [ ] **No direct `_` access** — Use public properties (wall, pattern_lines, floor_line)
- [ ] **No manual field mutation** — Use place(), process_round_end(), from_string()
- [ ] **No stale cached values** — Call `_encode()` if you manually change fields
- [ ] **No impossible states** — Ensure wall doesn't have duplicate colors per row
- [ ] **Setup uses public APIs** — place(), game.make_move(), game.advance()
- [ ] **Tests are readable** — State setup is clear; assertions test behavior, not implementation

---

## When Tests Break

**First check:**
1. Did you access `_pattern_grid`, `_wall`, or other removed private fields? → Use public properties
2. Did you manually set `player.score` or similar without calling `_encode()`? → Add `_encode()`
3. Did you manually set `_wall_tiles` or `_pattern_lines` directly? → Use place()
4. Are you getting `AssertionError` in `place()` about invalid tiles? → Check wall state isn't impossible

**Last resort:**
- Use `Game.from_string()` to build the exact state needed, without manual field access

---

## Testing from Game Recordings

**✅ Use Game.from_string() with actual game recordings:**

When reproducing bugs or testing edge cases, extract the game state from recorded games:

```python
from engine.game import Game
from engine.move import Move

# Load state from recording at a specific turn
game = Game.from_string("""
R1:T01 [1696940010]                               BAG 15 15 16 18 16
  P1: Player 1   0(  0)  > P2: Player 2   0(  0)  CLR  B  Y  R  K  W
          . | . . . . .            . | . . . . .  F-1  .  2  1  .  1
        . . | . . . . .          . . | . . . . .  F-2  .  .  .  .  .
      . B B | . . . . .        . . . | . . . . .  F-3  1  1  2  .  .
    . . . . | . . . . .      . . . . | . . . . .  F-4  1  1  .  .  2
  . . . . . | . . . . .    . . . . . | . . . . .  F-5  1  .  .  2  1
    ....... |                ....... |            CTR  .  1  1  .  .  F
""")

# Apply a move and test behavior
move = Move.from_str("2K-52")
game.players[1].place(move.destination, [move.tile] * move.count)
assert game.players[1].pending == 1  # Bug: was 4 before fix
```

**Benefits:**
- Reproduces real bugs from actual gameplay
- More realistic than synthetic test states
- Game state is parsed and guaranteed valid
- Excellent for regression testing

**See also:** `TEST_WRITER.md` "Loading Game State from Recording" section

---

## References

- **CODER.md** — Private fields pattern and `_encode()` requirement
- **TEST_WRITER.md** — Test organization, fixture patterns, and recording-based testing
- Engine refactor commit: ef8d8c5 (Player class refactoring)
- Integration fixes commit: 83968fb (47 test failures resolved)
- Pending scoring bug fix commit: 77fb82b (regression test for pending scoring)
