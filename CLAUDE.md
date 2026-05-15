# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) for working with this codebase and collaborating with the developer.

---

## How I Want to Work

- **Shorter responses** — We have Claude usage limits. If you get stuck, say so — we'll determine next steps or use another model.
- **TDD always** — Provide tests and implementation together, never after.
- **One step at a time** — Don't give me five files at once. Walk through each piece.
- **Explain the why** — If you make a design decision, explain it and any tradeoffs.
- **Git commits** — Suggest commit messages at natural stopping points (feature done, bug fixed, refactor complete). Don't remind after every edit.
- **Check CI** — We can proceed immediately after push, but CI might go red and we may have to backtrack.
- **Repeat on request** — If I ask for something already provided, just repeat without comment.
- **Sketch before coding** — For layout or visual changes, describe or sketch in text first to confirm agreement before writing code.
- **Branch for larger refactors** — Use feature branches (`feat/xxx`) for multi-commit work. Open draft PRs for CI on branches (CI runs on main and PRs only).
- **Don't shy away from refactoring** — Clean code allows continuing with less fear. Refactor when it improves the end product, simplifies code, or aligns with our mental model.
- **Budget-conscious** — Use read-only tools (Glob, Grep, Read, git history) heavily before suggesting changes. Batch related questions. Leverage existing patterns.

---

## Code Style Preferences

- **f-strings for logging and formatting** — Prefer `f"value: {x:.4f}"` over `"value: %.4f" % x`. Existing `%`-style calls stay unless already being edited.
- **Always provide full methods** — When editing a method, provide the complete new method.
- **Break long methods into sub-methods** — When a function exceeds ~20 lines or has multiple phases, split into named helpers. Prefer many small well-named functions over long ones with inline comments.
- **Descriptive helper names over inline complexity** — A helper named `_apply_warmup_floor_override(move, policy_pairs, game)` is more readable than a 15-line inline block with a comment.
- **Avoid abbreviations** — Well-named variables self-document. Short names only when immediately obvious and extremely small in scope.
- **Provide indented code** — Indent to the appropriate level for easy copy/paste into the IDE.

---

## Testing Preferences

- **TDD always** — Tests come with implementation, never after.
- **Test public methods only** — Private methods are covered implicitly by thorough public method tests.
- **Target 100% coverage in practice** — CI does not enforce a threshold, but we should hit it.
- **Test organization** — Files live in `tests/` subfolders mirroring source tree (`tests/engine/`, `tests/agents/`, etc.), each with `__init__.py`. Flat test files at `tests/` root stay until touched, then move to subfolders.

---

## Subagent Pattern

For large multi-file refactors or parallel work, consider spawning subagents. Each subagent handles one or a few files, keeping context small and enforcing separation of concerns.

**When to use:** 5+ files with low interdependency, or when files are large (500+ lines each).

**How it works:**
1. Main agent plans the refactor (explores codebase, understands architecture)
2. Spawns subagents, each with a focused prompt about their file(s)
3. Subagents work in parallel on independent changes
4. Main agent collects results and orchestrates merging/testing

**Overhead:** ~500–1000 tokens per subagent spawn. Pays off only for big refactors.

---

## Project Overview

**azul-alphazero** is an Azul board game engine (~7k LOC) implementing supervised learning with AlphaBeta AI. The game engine is pure Python; neural training uses PyTorch.

For architecture, design decisions, and roadmap, see **[docs/master_plan.md](docs/master_plan.md)**.

## Commands

All commands assume the `.venv` virtual environment is active.

### Testing

```bash
# Run all tests (fast tests only, excludes slow)
pytest

# Run a specific test file
pytest tests/engine/test_player.py

# Run a specific test class or function
pytest tests/engine/test_player.py::TestScoringProperties
pytest tests/engine/test_player.py::TestScoringProperties::test_earned_is_zero_for_fresh_player

# Run all tests including slow ones
pytest -m ""

# Show captured output during test runs
pytest -s
```

### Linting & Formatting

```bash
# Check for style/lint issues (flake8 with max-line 88, ignores E203, excludes scratch/)
flake8

# Format code with black (line length 88)
black .

# Sort imports with isort (black profile)
isort .
```

### Type Checking

Pylance is available in VS Code for real-time type hints. No separate type-checking CLI configured yet.

## Architecture

See **[docs/master_plan.md](docs/master_plan.md)** for the high-level vision, design decisions, and roadmap.

The codebase is organized into four main modules:

- **engine/** — Core game logic (game.py, player.py, constants.py)
- **agents/** — AI players implementing `Agent` base class (random, heuristic, search-based, neural)
- **neural/** — Deep learning (encoder, model, trainer, search tree, replay buffer)
- **tests/** — Scenario-based tests covering public APIs

## Key Development Patterns

### Encoding & Value Networks
- **engine/player.py** has `encode()` method and `ENCODING_SLICES` dict that defines sections (wall, pattern, score, penalties, etc.) for indexing into the 168-value encoding used by supervised learning.
- The feature branch `feat/supervised-value-net` extends value net encoding to 168 values (vs. the main model's 125-value flat encoding).

### Private Fields
Recent refactoring moved Player fields to private (e.g., `_wall`, `_pattern`). Properties like `wall` and `pattern` expose read-only views.

### Testing Conventions
- Test classes are organized by concern (TestScoringProperties, TestPolicyDistribution, etc.).
- Encoding tests use `Player.from_string()` to load known states, then assert sections of `encode()` output.
- Use `make_player(**kwargs)` helper to construct fresh players with optional field overrides.

### Move Format
Moves have a compact string representation: `{count}{tile}{marker}{source}{destination}` where marker is `-` (normal) or `+` (took first player). Parsed by `Move.from_string()`.

## Configuration

- **pyproject.toml** — Black (line length 88, target py312) and isort (black profile) config.
- **.flake8** — max-line-length 88, ignores E203, excludes scratch/ folder.
- **pytest.ini** — testpaths=tests, excludes slow tests by default (pytest -m "" to include).

## Critical Gotchas (Code Patterns)

These patterns have caused bugs before — keep them in mind:

- **Don't repeat the entire project state in docstrings** — Commit messages have the why. Code comments should explain non-obvious decisions (hidden constraints, workarounds, subtle invariants), not repeat what the code says.
- **Avoid premature abstraction** — Three similar lines is better than extracting a helper. Don't design for hypothetical future requirements.
- **No half-finished implementations** — If you start a feature, finish it. Don't leave stubs or "TODO" implementations.
- **Skip error handling for impossible scenarios** — Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs).
- **Test fixtures must be realistic** — Don't mock internal dependencies. Prefer hitting real objects (databases, game engine) so tests catch real breaks.

---

## Project-Specific Gotchas

See **[docs/alphabeta_strategy.md](docs/alphabeta_strategy.md#gotchas)** for Azul-specific patterns (round boundaries, advance() semantics, earned-score timing).

Key ones:
- `Move` uses `.tile`, not `.color`
- Always import `Tile` from `engine.constants`, never `engine.tile`
- **AlphaBeta searches only within round boundaries** — never call `advance()` without `skip_setup=True` inside tree search
- **Read `player.earned` before `advance()`** — after advance, bonus is folded into score and reading earned double-counts

---

## Notes for Future Sessions

- Scratch folder (scratch/) is excluded from flake8 and linting. Use it for exploratory work.
- Current active branch: `feat/supervised-value-net` (Player encoding refactor, Phase 0)
- For roadmap and design rationale, see **[docs/master_plan.md](docs/master_plan.md)**
