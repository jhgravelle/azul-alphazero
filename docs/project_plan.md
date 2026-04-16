# Azul AlphaZero — Project Plan

> Last updated: 2026-04-14
> Status: Phase 8 in progress.

---

## Vision

Build a fully playable implementation of the board game **Azul** with an **AlphaZero-style AI opponent**, deployable as a web app and eventually as a mobile app (iOS/Android).

---

## Technology Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | Primary language, best ML ecosystem |
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
│   └── alphazero.py       # Thin wrapper — delegates to SearchTree
├── neural/
│   ├── encoder.py         # (12,5,6) spatial + (47,) flat encoding
│   ├── model.py           # Conv+MLP hybrid: spatial branch + flat branch
│   ├── zobrist.py         # Zobrist hashing for within-round game states
│   ├── search_tree.py     # SearchTree: MCTS, transposition table, subtree reuse
│   ├── trainer.py
│   └── replay.py
├── api/
│   ├── main.py            # FastAPI app, persistent SearchTree per session
│   └── schemas.py
├── frontend/
│   ├── index.html
│   ├── render.js
│   ├── game.js
│   └── style.css
├── scripts/
│   ├── self_play.py
│   ├── train.py
│   └── migrate_recordings.py
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
│   ├── test_zobrist.py
│   ├── test_search_tree.py
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
- `AlphaZeroAgent` — thin wrapper around `SearchTree`
- `collect_self_play` — opponent=None (AZ vs AZ) or opponent=Agent (warmup mode)
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

- `earned_score(board)`, `grand_total(board)`, `pending_placement_details`, `pending_bonus_details`
- Score bar, wall annotations, sources row, game recorder, replay viewer, bag/box counts, full UI polish

---

### Phase 7 — Undo + Hypothetical + Manual Factory Setup ✅

#### 7a — Undo ✅
#### 7b — Hypothetical mode ✅
#### 7c — Manual factory setup ✅
#### 7d — Replay Improvements ✅

---

### Phase 8 — Evaluation and Iteration 🔄 (in progress)

#### 8a — Search and Encoding Rewrite ✅

Complete redesign of the neural network input encoding and MCTS search:

**Encoder (`neural/encoder.py`)**
- Spatial tensor `(12, 5, 6)` — 6 planes per player (5 color + 1 any-tile), rows = wall rows, cols 0–4 = wall cells, col 5 = pattern line fill ratio
- Flat vector `(47,)` — factories, center, tokens, floor lines, scores, bag, discard
- Move encoding unchanged: `(source, color, destination)` triple

**Model (`neural/model.py`)**
- Conv branch: two Conv2d layers over `(12, 5, 6)` spatial tensor
- MLP branch: linear projection of `(47,)` flat vector
- Branches merged before ResBlock trunk and policy/value heads

**Zobrist hashing (`neural/zobrist.py`)**
- Hashes only within-round state: pattern lines, floor lines, factories, center, current player
- Wall and scores excluded — frozen within a round

**SearchTree (`neural/search_tree.py`)**
- Game-owned, persists across turns within a round
- Transposition table: Zobrist hash → AZNode
- Subtree reuse: selected child becomes new root, siblings pruned
- Factory canonicalization: identical factories collapsed, reducing branching
- Round boundaries as leaf nodes: simulations stop at end-of-round, evaluated by value head

**AlphaZeroAgent (`agents/alphazero.py`)**
- Now a thin wrapper — owns an internal `SearchTree` for self-play/training contexts
- API passes in a shared external tree via `choose_move(game, tree=...)`
- Exposes `advance(move)` and `reset_tree(game)` for tree lifecycle management

**API (`api/main.py`)**
- Owns a persistent `SearchTree` at session level
- Tree advanced after every move, reset at every round boundary

#### 8b — Batched Multithreaded MCTS ✅

- Virtual loss to discourage thread collision on same path
- N threads collect leaves in parallel, single net forward pass per batch
- Single tree lock for thread safety (simple first implementation)
- Batch size and thread count as tunable parameters

#### 8c — Training Run + Iteration 🔄 (in progress)

**Completed**
- Rolling average bug confirmed fixed (regression test added)
- `advance_round_if_needed` added to `evaluate()` and `evaluate_vs_random()`
- Engine debug noise suppressed from training logs
- Eval recording saved each iteration to `recordings/eval/`
- Subfolder recording scan in API (`recordings/human/` and `recordings/eval/`)
- `_MAX_MOVES` raised to 2000
- Floor avoidance in warmup (`collect_self_play`) — prevents untrained model flooring everything
- Heuristic pretraining flags (`--pretrain-games`, `--pretrain-steps`, `--heuristic-iterations`)
- Loss diagnostic added — revealed policy vs value loss split

**Key finding**
Heuristic pretraining with one-hot policy targets causes policy head overfitting.
The value head learns score differentials quickly (~200 steps). The policy head
memorizes Greedy's exact moves rather than learning strategy. Fix: pretrain value
head only, let MCTS self-play train the policy from soft visit-count distributions.

**Next up**
- [ ] Implement `value_only=True` flag in `compute_loss` and `train_step`
- [ ] Update pretrain and heuristic iteration loops to use `value_only=True`
- [ ] Run training and verify self-play games complete without hitting move cap
- [ ] Elo ladder across all agent versions
- [ ] Hyperparameter search
- [ ] Difficulty levels in UI

---

### Phase 9 — Polish and Release
- [ ] Animated tile placement
- [ ] Sound effects
- [ ] Cloud deployment
- [ ] Capacitor iOS/Android packaging
- [ ] README with screenshots

### Future features discussed but not planned
- AlphaZero as UI opponent — wire into `_make_agent()` once a trained checkpoint exists
- Policy head annotations on hypothetical tree
- Multiple agent perspectives
- Shared-weight twin tower architecture for spatial encoding (considered, deferred)
- Two-phase moves for UI (considered, not needed for search)

---

## Agent Hierarchy

| Agent | Heuristics | Purpose |
|---|---|---|
| `RandomAgent` | None | Benchmark baseline |
| `CautiousAgent` | Floor-avoidance | Avoids penalties |
| `EfficientAgent` | Partial-line preference | Completes lines faster |
| `GreedyAgent` | Both heuristics | Default UI opponent |
| `MCTSAgent` | UCB1 + random rollouts | Lookahead without neural net |
| `AlphaZeroAgent` | PUCT + neural net via SearchTree | Final goal |

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
| 2026-04-07 | Phase 6b complete |
| 2026-04-07 | Phase 7 defined |
| 2026-04-13 | Phase 7 complete |
| 2026-04-13 | Phase 7d complete |
| 2026-04-14 | Phase 8a complete — spatial encoder, conv+MLP model, Zobrist hashing, SearchTree, AlphaZeroAgent refactor |
| 2026-04-14 | Phase 8b complete — batched multithreaded MCTS with virtual loss, thread pool backprop, tests |
| 2026-04-14 | Fixed advance_round_if_needed in eval loops, suppressed engine log noise, added eval recording per iteration, subfolder recording scan |
| 2026-04-15 | Phase 8c begun — fixed advance_round_if_needed in eval, floor avoidance in warmup, eval recordings, subfolder scan, raised move cap, loss diagnostic |