# TEST_WRITER.md

Test writing guidance and conventions.

## Before You Start

You're likely working from a plan. Read the plan completely and understand:
- What to test (which public methods, which scenarios)
- Test structure and naming conventions
- Coverage goals (aim for 100%)
- Success criteria for test completion

## Before Declaring Work Done

Run these checks. If any fail, fix the issues and re-run until all pass:

```bash
# Run all tests
python -m pytest tests/ -v

# Check linting on test files
python -m flake8 tests/

# Verify no test regressions in existing tests
# (failures here indicate code changes broke old functionality)
```

**Report:**
- ✅ All tests pass (X passed), linting clean, OR
- ❌ Issues found (list them), then re-check

If old tests break (not your new tests):
- Investigate why the code change broke them
- Fix the code (in CODER agent) or adjust the test if the behavior intentionally changed
- Loop back with feedback; don't declare done

Do NOT declare "done" if any tests fail or linting fails. Fix them first.

## Test Organization

- **Mirror source tree structure** — Test files live in `tests/` subfolders mirroring `src/` layout
  - Source: `engine/player.py` → Test: `tests/engine/test_player.py`
  - Source: `agents/random.py` → Test: `tests/agents/test_random.py`
- **Use `__init__.py`** — Each subfolder needs `__init__.py` (can be empty)
- **Keep root flat tests until touched** — Flat test files at `tests/` root stay there until modified, then move to subfolders

## Test Naming Conventions

- **Test classes** — Named after what they test: `TestScoringProperties`, `TestPolicyDistribution`, `TestGameInitialization`
- **Test methods** — Describe the scenario and expected outcome: `test_earned_is_zero_for_fresh_player`, `test_aimed_cells_exclude_pattern_lines`, `test_move_placement_respects_wall_tiles`
- **Fixtures** — Use descriptive names: `fresh_player`, `mid_game_state`, `completed_wall_row`

## What to Test

- **Test public methods only** — Private methods are covered implicitly by thorough public method tests
- **Organize by concern** — Use test classes named after what they test (e.g., `TestScoringProperties`, `TestPolicyDistribution`)
- **Target 100% coverage in practice** — CI does not enforce a threshold, but you should hit it
- **Test fixtures must be realistic** — Don't mock internal dependencies. Prefer hitting real objects (game engine, etc.) so tests catch real breaks

## Testing Patterns

### Using Test Fixtures

- **`make_player(**kwargs)`** — Construct fresh players with optional field overrides
- **`Player.from_string()`** — Load known game states for scenario-based tests
- **Encoding tests** — Use `Player.from_string()` to load state, then assert sections of `encode()` output

### Test Execution

See **COMMANDS.md** for how to run tests. Common patterns:

```bash
# Run all fast tests
pytest

# Run a specific test file
pytest tests/engine/test_player.py

# Run a specific test class
pytest tests/engine/test_player.py::TestScoringProperties

# Run all tests including slow ones
pytest -m ""

# Show captured output
pytest -s
```

## No Half-Finished Tests

- **Complete coverage of public API** — Don't leave TODOs or placeholder tests
- **Realistic data** — Use representative game states, not trivial edge cases only
- **Both happy path and edge cases** — Test normal behavior and boundary conditions
