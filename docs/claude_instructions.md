# Claude Instructions — Azul AlphaZero Project

---

## Project context

I am building an Azul board game engine with an AlphaZero AI in Python. This is a learning project — I am a beginner-to-intermediate Python developer. Please explain concepts as you go and don't skip steps.

**Tech stack:** Python 3.12, pytest, black, flake8, FastAPI + HTML/JS, PyTorch, Git + GitHub, GitHub Actions.

**Project structure and history:** See `PROJECT_PLAN.md` in the repo root.

---

## Current phase

> **Phase 8 — Evaluation and Iteration**

- Elo ladder across all agent versions
- Hyperparameter search
- Difficulty levels in UI

---

## How I want to work

- **TDD always:** Show me the test first. Let me read and understand it. Then show me the implementation.
- **One step at a time:** Don't give me five files at once. Walk me through each piece.
- **Explain the why:** If you make a design decision, tell me why, especially if there's a tradeoff.
- **Git commits:** Remind me when it's a good time to commit, and tell me what the commit message should be.
- **Check CI:** Remind me to push and check that GitHub Actions goes green before moving to the next piece.
- **Repeat on request:** If I ask for something you've already provided, just repeat it without comment.
- **Complete files for CSS:** When making CSS changes, always provide the complete file rather than incremental updates — partial CSS updates have caused sync issues in the past.
- **Sketch before coding:** For layout or visual changes, describe or sketch in text first to confirm we agree before writing any code.

---

## Critical gotchas

These have caused bugs before — always keep them in mind:

- `Move` uses `.tile`, not `.color`
- Always import `Tile` from `engine.constants`, never from `engine.tile`
- `_is_valid_destination` checks the specific wall column for that color in that row — not whether the color appears anywhere in the row
- `_score_floor` must filter out `Tile.FIRST_PLAYER` before adding to discard
- Empty `legal_moves()` mid-round is always a bug, never a valid game state
- `score_placement` precondition: tile must already be placed in the wall before calling
- **The API owns round transitions** — `_end_turn` scores the round but does not call `setup_round`; call `_handle_round_end()` after every `make_move` in the API
- **Simulation loops** (MCTS, AlphaZero, trainer) must call `game.advance_round_if_needed()` after every `make_move`
- **Recordings:** when adding fields to `GameRecord` or `RoundRecord`/`MoveRecord`, also update `scripts/migrate_recordings.py`
- **Windows:** use `python -m module.path` to run scripts, set `$env:PYTHONPATH = "."` if needed. Use `findstr` not `grep`. No Unicode in log strings.