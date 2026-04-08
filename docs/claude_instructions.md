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
- **Complete files for CSS:** When making CSS changes, always provide the complete file rather than incremental updates — partial CSS updates have caused sync issues in the past.
- **Sketch before coding:** For layout or visual changes, describe or sketch in text first to confirm we agree before writing any code.

---

## Current phase

> **Phase 7a — Undo with snapshot stack**

### What we are building

- `_history: list[GameState]` module-level variable in `api/main.py`
- Deep copy of `GameState` pushed onto `_history` before every `make_move` call in `/move` and `/agent-move`
- `POST /undo` endpoint — pops the last state from `_history` and restores it to `_game`
- Undo only available when at least one player is human (not bot-vs-bot)
- Undo button in the live game header — disabled when no history or bot-vs-bot
- `_history` clears on `POST /new-game`

After undo: hypothetical mode (7b), then manual factory setup (7c).

---

## Completed so far

### Phase 1 — Game Engine ✅

**`engine/constants.py`** — single source of truth for all fixed game data:
- `Tile` enum (BLUE, YELLOW, RED, BLACK, WHITE, FIRST_PLAYER)
- `COLOR_TILES` — list of the 5 non-marker tiles in enum order
- `BOARD_SIZE`, `PLAYERS`, `TILES_PER_COLOR`, `NUMBER_OF_FACTORIES`, `TILES_PER_FACTORY`
- `FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]`
- `WALL_PATTERN` — 5x5 grid, derived from `COLOR_TILES` and `BOARD_SIZE`
- `CUMULATIVE_FLOOR_PENALTIES` — indexed 0..NUMBER_OF_FACTORIES*TILES_PER_FACTORY
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
- `score_placement(wall, row, column)`
- `score_floor_penalty(floor_line)`
- `score_wall_bonus(wall)`
- `carried_score(board)`
- `earned_score(board)`
- `grand_total(board)`
- `pending_placement_details(board)` — returns list of detail objects + temp wall
- `pending_bonus_details(wall)` — returns list of bonus detail objects

**`engine/game_recorder.py`** — `GameRecorder`, `GameRecord`, `TurnRecord`:
- `record_turn(game, move, analysis=None)` — called BEFORE `make_move`
- `TurnRecord.source_state` captures: factories, center, bag_counts, discard_counts
- `TurnRecord.board_states` captures: wall, pattern_lines, floor_line, score per player
- `finalize(game)` — records final scores and winner
- `save(path)` / `GameRecord.load(path)` — JSON file I/O to `recordings/`

**`cli/cli.py`** — full terminal UI, human vs human.

**Known engine gotchas:**
- `_is_valid_destination` checks `player.wall[row][wall_column_for(row, tile)] is not None`
- `_score_floor` must filter out `Tile.FIRST_PLAYER` before adding to discard
- Empty `legal_moves()` mid-round is always a bug
- `score_placement` precondition: tile must be placed in wall before calling
- Always import `Tile` from `engine.constants`, never from `engine.tile`
- `Move` uses `.tile`, not `.color`

---

### Phase 2 — Graphical Front End ✅

**`api/schemas.py`:**
- `MoveRequest`, `NewGameRequest`, `PlayerType`
- `PendingPlacement`, `PendingBonus`, `BoardResponse`
- `GameStateResponse` — includes `bag_counts`, `discard_counts`, `round`
- `RecordingSummary` — for `GET /recordings` list

**`api/main.py`:**
- `GET /state`, `POST /move`, `POST /new-game`, `POST /agent-move`
- `GET /recordings`, `GET /recordings/{game_id}`
- `_recorder: GameRecorder | None` — created on `/new-game`, saved on game over
- `_RECORDINGS_DIR = Path("recordings")`
- `_build_response(game)` — translates engine state to `GameStateResponse`
- `_build_pending(board)` — computes pending placements and bonuses

**`frontend/index.html`** — single page, loads `render.js` then `game.js`, contains menu overlay HTML.

**`frontend/render.js`** — shared, no API calls:
- `makeTile(color, faint)` — adds `tile-placed` class for real tiles
- `makeInfoTile(color, count)` — for bag/box panel, uses `tile-info` class
- `renderSources(sources, opts)` — factories + center panel + bag/box panel
- `renderPatternLines(patternLines, opts)` — `droppable-row` class when interactive
- `renderFloorLine(floorLine, opts)` — rotated "Floor" label, penalties inside tiles
- `renderWall(wall, pendingPlacements, pendingBonuses)` — full wall with annotations
- `renderScoreDisplay(board)` — carried + pending + floor + bonus = total
- `renderBoard(board, index, label, isActive, opts)`
- `CENTER_SLOTS = 8` — placeholders before stacking kicks in

**`frontend/game.js`** — live game + replay mode + menu:
- `renderLive()` — full live game render
- `renderReplay()` — replay render using recorded snapshots
- `openMenu()` / `closeMenu()` / `initMenu()` — overlay menu
- `hasGameInProgress` — controls whether menu close button is enabled

**`frontend/style.css`** — full stylesheet:
- Tile states: `.tile` (placeholder), `.tile-placed` (shadow), `.tile-faint` (wall hint), `.tile-info` (bag/box)
- Selection: `.tile.selected` — white outline, no scale
- Droppable: `.droppable-row` — white outline on whole row
- All text overlays: white + `text-shadow: 0 1px 3px rgba(0,0,0,0.9)`
- Rotated labels: `.panel-label`, `.floor-label` — `writing-mode: vertical-rl; transform: rotate(180deg)`
- Pending tiles: full opacity, score overlaid as white text

---

### Phase 3 — Random Bot + Agent Interface ✅
### Phase 4 — Monte Carlo Tree Search ✅
### Phase 5 — Neural Network ✅
### Phase 6 — AlphaZero Self-Play Training 🔄 (paused)
### Phase 6b — Reward Shaping + UI Polish ✅

---

## Conventions

- black formatting + pre-commit hook installed
- isort, `extend-ignore = E203`
- Type hints on all signatures, docstrings on all public classes/functions
- American English: "tile", "center"
- **Never abbreviate `column` as `col` or `c`** — column and color are both core concepts
- **Prefer `tile` over `color` in variable names** where referring to a `Tile` enum value
- **Always import `Tile` from `engine.constants`**, never from `engine.tile`
- **`Move` uses `.tile`, not `.color`**
- Slow tests: `@pytest.mark.slow`, excluded by default
- `checkpoints/` and `recordings/` are gitignored
- Never use `print()` in engine code — use `logging`
- Never import `api/` or `frontend/` from `engine/`
- No Unicode characters in log strings — use plain ASCII only (Windows console encoding)
- **Windows:** use `python -m module.path` to run scripts, set `$env:PYTHONPATH = "."` if needed. Use `findstr` not grep.
- **CSS:** always provide complete files, not incremental updates
- **Layout/visual changes:** sketch in text first before writing code