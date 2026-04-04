# Azul AlphaZero — Project Plan

> Last updated: 2026-04-04
> Status: Phase 6b Step 1 complete — engine scoring functions done. Step 2 (API) up next.

---

## Vision

Build a fully playable implementation of the board game **Azul** with an **AlphaZero-style AI opponent**, deployable as a web app and eventually as a mobile app (iOS/Android).

---

## Technology Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.14 | Primary language, best ML ecosystem |
| Front end | FastAPI + HTML/JS | Web-first, iPhone-compatible via Capacitor, shareable by URL |
| Testing | pytest | Industry standard for Python TDD |
| Version control | Git + GitHub | Standard, CI/CD integration |
| CI/CD | GitHub Actions | Free for public repos, integrates natively with GitHub |
| ML framework | PyTorch | Best for custom AlphaZero-style training loops |
| IDE | VS Code | Installed, good Python + git support |

---

## Architecture Overview

```
azul-alphazero/
├── engine/
│   ├── constants.py      # Tile enum, COLORS, WALL_PATTERN, precomputed lookups
│   ├── board.py
│   ├── game_state.py
│   ├── game.py
│   ├── scoring.py        # Pure scoring functions
│   └── factory.py
├── agents/
│   ├── base.py
│   ├── random.py
│   ├── cautious.py
│   ├── efficient.py
│   ├── greedy.py
│   ├── mcts.py
│   └── alphazero.py
├── neural/
│   ├── encoder.py
│   ├── model.py
│   ├── trainer.py
│   └── replay.py
├── api/
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── index.html
│   ├── game.js
│   └── style.css
├── scripts/
│   ├── self_play.py
│   └── train.py
├── tests/
│   ├── test_constants.py
│   ├── test_game.py
│   ├── test_board.py
│   ├── test_scoring.py
│   ├── test_agents.py
│   ├── test_api.py
│   ├── test_self_play.py
│   ├── test_mcts.py
│   ├── test_encoder.py
│   ├── test_model.py
│   ├── test_replay.py
│   ├── test_trainer.py
│   └── test_alphazero.py
├── checkpoints/     # gitignored
└── docs/
    └── PROJECT_PLAN.md
```

---

## Development Phases

### Phase 0 — Project Setup ✅
### Phase 1 — Game Engine ✅
### Phase 2 — Graphical Front End ✅
### Phase 3 — Random Bot + Agent Interface ✅
### Phase 4 — Monte Carlo Tree Search ✅
### Phase 5 — Neural Network ✅

---

### Phase 6 — AlphaZero Self-Play Training 🔄 (paused for 6b)

#### What's built
- `AlphaZeroAgent` — PUCT tree search, value head evaluation, no rollouts
- `collect_self_play` — opponent=None (AZ vs AZ) or opponent=Agent (warmup mode)
- `collect_heuristic_games` — 50% Greedy, 25% Cautious, 25% Efficient, one-hot policy targets
- `scripts/train.py` — full training loop with greedy warmup, auto-switch, per-game eval logging, `_MAX_MOVES=300`

#### Known issues to fix before next training run
- Rolling avg bug: records 0 for AZ-as-p1 games → warmup threshold never reached
- 100 train steps too few — increase to 500
- 20 eval games too noisy — increase to 40 or lower threshold to 0.48

#### Remaining tasks
- [ ] Fix rolling average bug in `collect_self_play`
- [ ] Increase train steps, tune eval threshold
- [ ] Wire best checkpoint into API `_make_agent()`
- [ ] Add AlphaZero as UI opponent option
- [ ] Elo rating system

---

### Phase 6b — Reward Shaping 🔄 (in progress)

**Motivation:** Azul's scoring is highly deferred. The value head has no signal until end of round or end of game. Moving the reward signal closer to the move that earned it should dramatically accelerate learning.

#### Engine (scoring.py) ✅

**`carried_score(board) -> int`** — `board.score`. Named accessor for the four-part model.

**`score_floor_penalty(floor_line) -> int`** — penalty for current floor tiles. Uses `CUMULATIVE_FLOOR_PENALTIES` lookup.

**`score_placement(wall, row, column) -> int`** — score a single tile placement. Precondition: tile already placed in wall before calling. Uses pointer-walk for performance.

**`score_wall_bonus(wall) -> int`** — end-of-game bonuses (+2 row, +7 column, +10 color) for tiles already on the wall.

**`earned_score(board) -> int`** — points earned this round not yet in `board.score`. Simulates pending pattern line placements sequentially (row 0 first) on a temporary wall copy, so adjacency between pending placements is captured correctly. Includes floor penalties and wall bonuses on the post-placement wall.

**`grand_total(board) -> int`** — `carried_score + earned_score`. Not clamped — can be negative.

#### Precomputed lookups in constants.py ✅
- `WALL_PATTERN` — moved here from `game.py`
- `CUMULATIVE_FLOOR_PENALTIES` — indexed 0..NUMBER_OF_FACTORIES*TILES_PER_FACTORY, no capping needed
- `COLUMN_FOR_COLOR_IN_ROW[tile][row]` — replaces all `.index()` calls on the wall pattern

#### API — expose scoring breakdown per player (up next)

Add to `BoardResponse` (schemas.py):
- `carried_score: int`
- `earned_score: int`
- `bonus_score: int`
- `grand_total: int`

`bonus_score` is the wall bonus component of `earned_score` broken out separately for the UI.

#### UI (after API)
- Wall tile preview: show `+N` on wall cell where a full pattern line will score
- End-of-game bonus indicators: `+7` below completed columns, `+10` for completed colors, `+2` right of completed rows
- Four-part score display: Carried | Earned | Bonus | Total

#### Model integration (after UI)
- Replace final-game-score value target in `collect_self_play` with `grand_total` delta per move
- Model receives only `grand_total` — no breakdown

---

### Phase 7 — Evaluation and Iteration
- [ ] Elo ladder across all agent versions
- [ ] Hyperparameter search
- [ ] Difficulty levels in UI

### Phase 8 — Polish and Release
- [ ] Animated tile placement
- [ ] Sound effects
- [ ] Game history / move replay
- [ ] Cloud deployment
- [ ] Capacitor iOS/Android packaging
- [ ] README with screenshots

---

## Agent Hierarchy

| Agent | Heuristics | Purpose |
|---|---|---|
| `RandomAgent` | None | Benchmark baseline |
| `CautiousAgent` | Floor-avoidance | Avoids penalties |
| `EfficientAgent` | Partial-line preference | Completes lines faster |
| `GreedyAgent` | Both heuristics | Default UI opponent |
| `MCTSAgent` | UCB1 + random rollouts | Lookahead without neural net |
| `AlphaZeroAgent` | PUCT + neural net | Final goal |

---

## Key Principles

**TDD always.** Engine independence. Commit often. CI is the source of truth.

---

## Change Log

| Date | Change |
|---|---|
| 2026-03-29 | Initial plan |
| 2026-04-01 | Phases 1-3 complete |
| 2026-04-01 | Phase 4 complete |
| 2026-04-02 | Phase 5 complete |
| 2026-04-02 | Phase 6 in progress |
| 2026-04-03 | Phase 6 run 4 complete — failure analysis, reward shaping planned |
| 2026-04-03 | Phase 6b defined |
| 2026-04-04 | Phase 6b Step 1 complete — scoring.py, constants.py refactor, game.py cleanup |