# Claude Instructions — Azul AlphaZero Project

---

## Project context

I am building an Azul board game engine with an AlphaZero AI in Python. This is a learning project — I am a beginner-to-intermediate Python developer. Please explain concepts as you go and don't skip steps.

**Tech stack:** Python 3.14, pytest, black, flake8, FastAPI + HTML/JS, PyTorch, Git + GitHub, GitHub Actions.

**Project structure:** See `PROJECT_PLAN.md` in the repo root.

---

## How I want to work

- **TDD always:** Show me the test first. Let me read and understand it. Then show me the implementation.
- **One step at a time:** Don't give me five files at once. Walk me through each piece.
- **Explain the why:** If you make a design decision, tell me why, especially if there's a tradeoff.
- **Git commits:** Remind me when it's a good time to commit, and tell me what the commit message should be.
- **Check CI:** Remind me to push and check that GitHub Actions goes green before moving to the next piece.
- **Repeat on request:** If I ask for something you've already provided, just repeat it without comment.

---

## Current phase

> **Phase 6b Step 2 — Expose scoring breakdown in the API**

### What we are building

Add `carried_score`, `earned_score`, `bonus_score`, and `grand_total` per player to `BoardResponse` in `api/schemas.py`, and populate them in `api/main.py`.

`bonus_score` is `score_wall_bonus(board.wall)` — the wall bonus component of `earned_score` broken out separately so the UI can display it distinctly.

After the API step: UI changes (wall tile preview annotations, bonus indicators, four-part score display), then model integration.

---

## Completed so far

### Phase 1 — Game Engine ✅

**`engine/constants.py`** — single source of truth for all fixed game data:
- `Tile` enum (BLUE, YELLOW, RED, BLACK, WHITE, FIRST_PLAYER)
- `COLOR_TILES` — list of the 5 non-marker tiles in enum order
- `BOARD_SIZE`, `PLAYERS`, `TILES_PER_COLOR`, `NUMBER_OF_FACTORIES`, `TILES_PER_FACTORY`
- `FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]`
- `WALL_PATTERN` — 5x5 grid, derived from `COLOR_TILES` and `BOARD_SIZE`
- `CUMULATIVE_FLOOR_PENALTIES` — indexed 0..NUMBER_OF_FACTORIES*TILES_PER_FACTORY, no capping needed at call site
- `COLUMN_FOR_COLOR_IN_ROW[tile][row]` — precomputed wall column lookup

**`engine/tile.py`** — re-exports `Tile` and `COLOR_TILES` from `constants.py` for backwards compatibility only. All new code imports from `engine.constants` directly.

**`engine/board.py`** — `Board` dataclass: `score`, `pattern_lines`, `wall`, `floor_line`.

**`engine/game_state.py`** — `GameState` dataclass.

**`engine/game.py`** — `WALL_PATTERN` re-exported for tests, `Game` class:
- `setup_round`, `legal_moves`, `make_move`, `score_round`, `is_game_over`, `score_game`
- `wall_column_for` uses `COLUMN_FOR_COLOR_IN_ROW`
- `_score_floor` uses `CUMULATIVE_FLOOR_PENALTIES`
- `score_round` calls `score_placement` from `scoring.py`
- `score_game` calls `score_wall_bonus` from `scoring.py`

**`engine/scoring.py`** — pure scoring functions (no state mutation):
- `score_placement(wall, row, column)` — pointer-walk, precondition: tile already placed
- `score_floor_penalty(floor_line)` — single lookup into `CUMULATIVE_FLOOR_PENALTIES`
- `score_wall_bonus(wall)` — +2/+7/+10 bonuses for completed rows/columns/colors
- `carried_score(board)` — returns `board.score`
- `earned_score(board)` — simulates pending placements sequentially on temp wall copy, includes floor penalty and wall bonus on post-placement wall
- `grand_total(board)` — `carried_score + earned_score`, not clamped

**`cli/cli.py`** — full terminal UI, human vs human, colored tiles, dim wall hints.

**Known engine gotchas:**
- `_is_valid_destination` checks `player.wall[row][wall_column_for(row, tile)] is not None` — not `tile in player.wall[row]`
- `_score_floor` must filter out `Tile.FIRST_PLAYER` before adding to discard
- Empty `legal_moves()` mid-round is always a bug, not an edge case
- `score_placement` precondition: tile must be placed in wall before calling

---

### Phase 2 — Graphical Front End ✅

- `api/schemas.py` — `MoveRequest`, `BoardResponse`, `GameStateResponse`, `NewGameRequest`, `PlayerType`
- `api/main.py` — GET /state, POST /move, POST /new-game, POST /agent-move, `_make_agent()`
- `frontend/` — full click-to-move UI, New Game dialog, bot turns via `maybeRunBot()`, 2s inter-round pause

---

### Phase 3 — Random Bot + Agent Interface ✅

- `agents/base.py` — abstract `Agent` with `choose_move(game) -> Move`
- `agents/random.py`, `cautious.py`, `efficient.py`, `greedy.py`
- `scripts/self_play.py` — `run_game`, `run_series`, `AGENT_REGISTRY`

---

### Phase 4 — Monte Carlo Tree Search ✅

- `agents/mcts.py` — UCB1, `MCTSNode`, `_select/_expand/_simulate/_backpropagate`

---

### Phase 5 — Neural Network ✅

- `neural/encoder.py` — `encode_state` (116 floats), `encode_move`, `decode_move`, `STATE_SIZE=116`, `MOVE_SPACE_SIZE=180`
- `neural/model.py` — `AzulNet`: stem + 3×ResBlock(256) + policy + value heads
- `neural/replay.py` — `ReplayBuffer`: circular buffer, push/sample
- `neural/trainer.py` — `compute_loss`, `Trainer`, `collect_self_play`, `collect_heuristic_games`

---

### Phase 6 — AlphaZero Self-Play Training 🔄 (paused for 6b)

**`agents/alphazero.py`:**
- `AZNode` dataclass, PUCT selection (C=1.5), expand/evaluate/backpropagate
- `_evaluate` — value head only, no rollouts
- `get_policy_targets` — visit-count distribution for training
- temperature: 0.0 = greedy, 1.0 = proportional

**`neural/trainer.py`:**
- `collect_self_play(buf, net, num_games, simulations, temperature, opponent)`
- `collect_heuristic_games(buf, num_games)`

**`scripts/train.py`:**
- `--pretrain-games`, `--greedy-warmup`, `--warmup-threshold`, `--warmup-window`
- Per-game eval logging, `_MAX_MOVES=300`, reset-to-best on failed threshold

**Known issues to fix before next training run:**
- Rolling avg bug: records 0 for AZ-as-p1 games → warmup threshold never reached
- 100 train steps too few — increase to 500
- 20 eval games too noisy — increase to 40 or lower threshold to 0.48

---

### Phase 6b Step 1 — Engine scoring functions ✅

See `engine/scoring.py` and `engine/constants.py` above.

---

## Conventions

- black formatting, isort, `extend-ignore = E203`
- Type hints on all signatures, docstrings on all public classes/functions
- American English: "tile", "center"
- **Never abbreviate `column` as `col` or `c` in variable names** — column and color are both core game concepts and must be unambiguous
- **Prefer `tile` over `color` in variable names** where referring to a `Tile` enum value
- Slow tests: `@pytest.mark.slow`, excluded by default
- `checkpoints/` is gitignored
- Never use `print()` in engine code — use `logging`
- Never import `api/` or `frontend/` from `engine/`
- Never skip writing tests
- When writing a new test file, give the most basic implementation file to get past import errors
- No Unicode characters in log strings — use plain ASCII only (Windows console encoding)
- **Windows:** use `python -m module.path` to run scripts, set `$env:PYTHONPATH = "."` if needed. Do not suggest grep — use `findstr` or VS Code search instead