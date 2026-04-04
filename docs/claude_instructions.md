# Claude Instructions — Azul AlphaZero Project

Paste this into your Claude project instructions so every conversation starts with the right context.

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

> **Phase 6b — Reward Shaping (up next)**

### What we are building

Two new engine methods, plus UI changes, plus model integration:

**`carried_score(board)`** — the official score carried from end of last round. Equals `board.score`. Named accessor for clarity.

**`earned_score(board, wall_pattern)`** — points earned this round but not yet received:
- Wall placement scores for all currently full pattern lines
- Floor penalties for tiles currently on the floor line
- End-of-game bonuses for completed rows/columns/colors already on the wall

**`grand_total(board, wall_pattern)`** — `carried_score + earned_score`

These belong in `engine/scoring.py`. They are pure game logic, not training artifacts.

### UI changes planned
- Wall tile preview: show `+N` on wall cell where a full pattern line will score. Refreshes as adjacencies change.
- End-of-game bonus indicators: `+7` below completed columns, `+10` centered below completed color's last tile, `+2` right of completed rows
- Four-part score display: Carried | Earned | Bonus | Total

### Model integration (after UI)
- Replace final-game-score value target in `collect_self_play` with `grand_total` delta per move
- Model receives only `grand_total` — no breakdown — must learn that increases are good

### Sequencing
1. Engine methods + tests first
2. UI second
3. Model integration third

---

## Completed so far

### Phase 1 — Game Engine ✅

- `tile.py` — Tile enum, COLORS list
- `constants.py` — BOARD_SIZE, PLAYERS, TILES_PER_COLOR, TILES_PER_FACTORY, FLOOR_PENALTIES
- `board.py` — Board dataclass (pattern lines, wall, floor line, score)
- `game_state.py` — GameState dataclass
- `game.py` — WALL_PATTERN, wall_column_for, setup_round, legal_moves, _is_valid_destination, make_move, score_round, is_game_over, score_game
- `cli/cli.py` — full terminal UI, human vs human, colored tiles, dim wall hints

**Known engine gotchas:**
- `_is_valid_destination` checks `player.wall[row][wall_column_for(row, color)] is not None` — not `color in player.wall[row]`
- `_score_floor` must filter out `Tile.FIRST_PLAYER` before adding to discard
- Empty `legal_moves()` mid-round is always a bug, not an edge case

---

### Phase 2 — Graphical Front End ✅

- `api/schemas.py` — MoveRequest, BoardResponse, GameStateResponse, NewGameRequest, PlayerType
- `api/main.py` — GET /state, POST /move, POST /new-game, POST /agent-move, _make_agent()
- `frontend/` — full click-to-move UI, New Game dialog, bot turns via maybeRunBot(), 2s inter-round pause

---

### Phase 3 — Random Bot + Agent Interface ✅

- `agents/base.py` — abstract Agent with `choose_move(game) -> Move`
- `agents/random.py`, `cautious.py`, `efficient.py`, `greedy.py`
- `scripts/self_play.py` — run_game, run_series, AGENT_REGISTRY

---

### Phase 4 — Monte Carlo Tree Search ✅

- `agents/mcts.py` — UCB1, MCTSNode, _select/_expand/_simulate/_backpropagate

---

### Phase 5 — Neural Network ✅

- `neural/encoder.py` — encode_state (116 floats), encode_move, decode_move, STATE_SIZE=116, MOVE_SPACE_SIZE=180
- `neural/model.py` — AzulNet: stem + 3×ResBlock(256) + policy + value heads
- `neural/replay.py` — ReplayBuffer: circular buffer, push/sample
- `neural/trainer.py` — compute_loss, Trainer, collect_self_play, collect_heuristic_games

---

### Phase 6 — AlphaZero Self-Play Training 🔄 (in progress, paused for 6b)

**`agents/alphazero.py`:**
- AZNode dataclass, PUCT selection (C=1.5), expand/evaluate/backpropagate
- `_evaluate` — value head only, no rollouts
- `get_policy_targets` — visit-count distribution for training
- temperature: 0.0 = greedy, 1.0 = proportional

**`neural/trainer.py`:**
- `collect_self_play(buf, net, num_games, simulations, temperature, opponent)`
  - opponent=None → AZ vs AZ; opponent=Agent → warmup mode
  - Returns list[int] of AZ scores for rolling average
- `collect_heuristic_games(buf, num_games)` — 50/25/25 Greedy/Cautious/Efficient mix

**`scripts/train.py`:**
- `--pretrain-games`, `--greedy-warmup`, `--warmup-threshold`, `--warmup-window`
- Per-game eval logging, `_MAX_MOVES=300`, reset-to-best on failed threshold

**Known issues to fix before next training run:**
- Rolling avg bug: records 0 for AZ-as-p1 games → warmup threshold never reached
- 100 train steps too few — increase to 500
- 20 eval games too noisy — increase to 40 or lower threshold to 0.48

**Training history:**

| Run | Result |
|---|---|
| Test run | Gen 0001: 55% vs prev, 32.5% vs random |
| Overnight 1 | Interrupted by Windows sleep |
| Overnight 2 | Regressed after iter 1 — [0,0] buffer poisoning |
| Run 4 (160 iter) | 1 generation only — rolling avg bug, never switched to self-play |

**Windows sleep prevention:**
```powershell
powercfg /requestsoverride process python.exe system
# restore after:
powercfg /requestsoverride process python.exe
```

---

## Conventions

- black formatting, isort, `extend-ignore = E203`
- Type hints on all signatures, docstrings on all public classes/functions
- American English: "color", "center"
- Slow tests: `@pytest.mark.slow`, excluded by default
- `checkpoints/` is gitignored
- Never use `print()` in engine code — use `logging`
- Never import `api/` or `frontend/` from `engine/`
- Never skip writing tests
- No Unicode characters in log strings — use plain ASCII only (Windows console encoding)