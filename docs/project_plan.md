# Azul AlphaZero — Project Plan

> Last updated: 2026-04-13
> Status: Phase 7 complete. Phase 8 (Evaluation and Iteration) up next.

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
| IDE | VS Code + Claude Code | Installed, good Python + git support |

---

## Architecture Overview

```
azul-alphazero/
├── engine/
│   ├── constants.py       # Tile enum, WALL_PATTERN, FLOOR_PENALTIES, all constants
│   ├── board.py           # Board dataclass
│   ├── game_state.py      # GameState dataclass
│   ├── game.py            # Game controller: make_move, legal_moves, score_round, etc.
│   ├── scoring.py         # Pure scoring functions
│   └── game_recorder.py   # GameRecorder, GameRecord, RoundRecord, MoveRecord
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
│   ├── main.py            # FastAPI app, endpoints, recorder integration
│   └── schemas.py         # Pydantic request/response models
├── frontend/
│   ├── index.html         # Single page — live game + replay
│   ├── render.js          # Shared rendering functions (no API calls)
│   ├── game.js            # Live game logic + replay mode + menu
│   └── style.css
├── scripts/
│   ├── self_play.py
│   ├── train.py
│   └── migrate_recordings.py  # Migrates old recordings to current format
├── tests/
│   ├── test_tile.py
│   ├── test_board.py
│   ├── test_game.py
│   ├── test_game_state.py
│   ├── test_scoring.py
│   ├── test_game_recorder.py
│   ├── test_api.py
│   ├── test_agents.py
│   ├── test_mcts.py
│   ├── test_encoder.py
│   ├── test_model.py
│   ├── test_replay.py
│   ├── test_trainer.py
│   └── test_alphazero.py
├── recordings/            # gitignored — one JSON per completed human game
├── checkpoints/           # gitignored
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

### Phase 6 — AlphaZero Self-Play Training 🔄 (paused)

#### What's built
- `AlphaZeroAgent` — PUCT tree search, value head evaluation, no rollouts
- `collect_self_play` — opponent=None (AZ vs AZ) or opponent=Agent (warmup mode); records both players
- `collect_heuristic_games` — Greedy vs Random; skips Random-wins games
- `scripts/train.py` — full training loop with greedy warmup, auto-switch, per-game eval logging

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

### Phase 6b — Reward Shaping + UI Polish ✅

- `earned_score(board)` — simulates pending placements, includes floor penalty and wall bonus
- `grand_total(board)` — carried_score + earned_score
- `pending_placement_details(board)` — per-cell placement scores for UI annotations
- `pending_bonus_details(wall)` — completed row/column/color bonuses for UI
- Score bar: carried + pending placements + floor penalty + bonuses = grand total
- Wall annotations: `+N` on pending placement cells, row/column/color bonus indicators
- Sources row: factories + Center panel + Bag/Box panel
- Game recorder, replay viewer, bag/box counts, full UI polish pass

---

### Phase 7 — Undo + Hypothetical + Manual Factory Setup ✅

#### 7a — Undo ✅
- `_history: list[GameState]` in `api/main.py`
- Deep copy pushed before every `make_move`
- `POST /undo` — pops and restores; automatically skips through bot moves to land on human turn
- Disabled in bot-vs-bot games

#### 7b — Hypothetical mode ✅
- "What if?" button overrides both players to human
- Hypothetical tree panel — branching, node jumping, commit execution
- From-replay hypothetical entry
- Terminal states in hypothetical are leaf nodes — no round setup, no recording saved

#### 7c — Manual factory setup ✅
- Pre-game step: human clicks tiles into each factory
- `POST /setup-factories/*` endpoints
- Persists across all rounds of the game — `_handle_round_end` re-enters setup mode
- First-player marker correctly placed in center on setup entry

#### 7d — Replay Improvements ✅
- **Compact recording format** — rounds/moves instead of full board snapshots per turn
- **`GameRecord.reconstruct()`** — replays moves server-side, embeds `computed_turns` and `final_boards` in API response; `computed_turns[0]` is always the initial state
- **Move list panel** — below boards; round headers (Round 1, 2…); turn numbers; player emoji (👤/🤖); tile chip; source→destination; grand totals; scroll-to-current; keyboard navigation (arrow keys)
- **Grand totals in replay** — boards show earned scores immediately, not end-of-round scored values; pending placements and bonuses computed during reconstruction
- **Auto-load replay** — game automatically transitions to replay mode 1500ms after game over using `last_game_id` in `GameStateResponse`
- **P1/P2 labels** — player boards labeled `P1 Human`, `P2 Greedy Bot` etc. in live game; recordings store prefixed names
- **Human-readable recording filenames** — `YYYYMMDD HHMMSS P1 name score - P2 name score.json`
- **Migration script** — `scripts/migrate_recordings.py` converts old verbose format to compact format; detects round boundaries from factory state; adds P1/P2 prefixes; idempotent with .bak backups

---

### Phase 8 — Evaluation and Iteration 🔜 (up next)

- [ ] Elo ladder across all agent versions
- [ ] Hyperparameter search
- [ ] Difficulty levels in UI

### Phase 9 — Polish and Release
- [ ] Animated tile placement
- [ ] Sound effects
- [ ] Cloud deployment
- [ ] Capacitor iOS/Android packaging
- [ ] README with screenshots

### Future features discussed but not planned
- AlphaZero as UI opponent — once a trained checkpoint exists, wire into `_make_agent()`
- Policy head annotations on hypothetical tree — show each move's prior probability and value estimate
- Multiple agent perspectives — annotate same position with evaluations from different checkpoints
- Bot moves in hypothetical tree show no move label — fixable by diffing pre/post state or having API return last move made

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
| 2026-04-03 | Phase 6b defined — carried_score, earned_score, grand_total, UI display |
| 2026-04-07 | Phase 6b complete — recorder, replay viewer, bag/box counts, full UI polish pass |
| 2026-04-07 | Phase 7 defined — undo, hypothetical mode, manual factory setup |
| 2026-04-13 | Phase 7 complete — undo, hypothetical, manual factories, replay improvements |
| 2026-04-13 | Phase 7d complete — compact recording format, move list panel, grand totals in replay, auto-load replay, P1/P2 labels, migration script |