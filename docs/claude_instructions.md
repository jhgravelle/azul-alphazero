# Claude Instructions — Azul AlphaZero Project

---

## Project context

I am building an Azul board game engine with an AlphaZero AI in Python. This is a learning project — I am a beginner-to-intermediate Python developer. Please explain concepts as you go and don't skip steps.

**Tech stack:** Python 3.12, pytest, black, flake8, FastAPI + HTML/JS, PyTorch, Git + GitHub, GitHub Actions, Windows 11

**Full project state:** See `docs/project_plan.md`.

---

## How I want to work

- **Shorter responses:** We have Claude usage limits. If you get stuck, say so — we will determine where to go next or craft a prompt for another model.
- **TDD always:** Provide the test and implementation at the same time.
- **One step at a time:** Don't give me five files at once. Walk me through each piece.
- **Explain the why:** If you make a design decision, say why — especially if there is a tradeoff.
- **Git commits:** Tell me the commit message at a natural stopping point — after a feature is complete, a bug is fixed, or a refactor is done. Don't remind me after every single edit.
- **Check CI:** We can proceed immediately after push, but CI might go red and we may have to backtrack.
- **Repeat on request:** If I ask for something you have already provided, just repeat it without comment.
- **Sketch before coding:** For layout or visual changes, describe or sketch in text first to confirm we agree before writing any code.
- **Branch for larger refactors:** Use feature branches (`feat/xxx`) and draft PRs for work that spans multiple commits. CI only runs on `main` and PRs, so opening a draft PR is how to get CI on a feature branch.
- **Don't shy away from refactoring:** Clean code allows us to continue making changes with less fear of breaking something. Refactor when it improves the end product, simplifies code, or aligns better with our mental model.

---

## Code style preferences

- **f-strings for logging and string formatting.** Prefer `f"something: {value:.4f}"` over `"something: %.4f" % value`. Existing `%`-style logging calls can stay as-is unless already being edited.
- **Always provide full methods:** When editing a method, always provide the new complete method.
- **Break long methods into sub-methods.** When a function grows past ~20 lines or has multiple distinct phases, split it into named helpers. Prefer many small well-named functions over long ones with inline comments marking sections.
- **Descriptive helper names over inline complexity.** A helper named `_apply_warmup_floor_override(move, policy_pairs, game)` is more readable than a 15-line inline block with a comment.
- **Avoid abbreviations.** Well-named variables self-document the code. Short variable names only when immediately obvious to a beginner programmer and extremely small in scope.
- **Provide indented code.** Indent code to the appropriate level for easier copy/paste into the IDE.

---

## Testing preferences

- **TDD always** — tests come with the implementation, never after.
- **Test public methods only.** Private methods are covered implicitly by thorough public method tests.
- **Target 100% coverage** in practice. CI does not enforce a threshold, but we should hit it.
- **Test files** live in `tests/` subfolders mirroring the source tree (`tests/engine/`, `tests/neural/`, etc.), each with an `__init__.py`. Flat test files at `tests/` root stay put until touched, then move to the appropriate subfolder.

---

## Critical gotchas

These have caused bugs before — always keep them in mind:

- `Move` uses `.tile`, not `.color`
- Always import `Tile` from `engine.constants`, never from `engine.tile`
- `_is_valid_destination` checks the specific wall column for that color in that row — not whether the color appears anywhere in the row
- `_score_floor` must filter out `Tile.FIRST_PLAYER` before adding to discard
- Empty `legal_moves()` mid-round is always a bug, never a valid game state
- `score_placement` precondition: tile must already be placed in the wall before calling
- **The API owns round transitions** — `_handle_round_end()` must be called after every `make_move` in the API; never call `advance()` directly in API code
- **Simulation loops** (MCTS, AlphaZero, trainer) must call `game.advance()` after every `make_move`
- **Minimax/AlphaBeta** must call `advance(skip_setup=True)`; compute scores using `player.earned` BEFORE advance — advance mutates the wall and resets scoring caches
- **Recordings:** when adding fields to `GameRecord` or `RoundRecord`/`MoveRecord`, also update `scripts/migrate_recordings.py`
- **Windows:** use `python -m module.path` to run scripts, set `$env:PYTHONPATH = "."` if needed. Use `findstr` not `grep`. No Unicode in log strings.
- **Adding a new agent** requires: `agents/registry.py` entry, `api/schemas.py` PlayerType update. These must be two separate locations due to circular import constraints — investigate consolidation when the time comes.