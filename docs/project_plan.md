# Azul AlphaZero — Project Plan

> Last updated: 2026-04-22
> Status: Phase 8d complete. Encoding upgraded: blocked_wall channel, earned_score_unclamped, round progress, distinct source-color pairs. Ready for training run with AlphaBeta as pretrain and eval opponent.

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
| OS | Windows 11 25H2 64-bit | What I have |
| GPU | nVidia RTX 5070 | What I have |
| CPU | Ryzen 7 7800X3d 8-Core | What I have |

**Compute split**: MCTS inference runs on CPU (faster for small batch sizes and this model), training runs on GPU. GPU sits near-idle during self-play and eval; CPU is the actual bottleneck.

---

## Architecture Overview

```
azul-alphazero/
├── agents/
│   ├── alphazero.py       # Thin wrapper — delegates to SearchTree
│   ├── alphabeta.py       # Alpha-beta pruning with cheap move ordering
│   ├── base.py            # Agent base class + default policy_distribution (uniform)
│   ├── cautious.py        # Uniform over non-floor moves
│   ├── efficient.py       # Uniform over partial-line moves (fallback to all)
│   ├── greedy.py          # Color-conditional distribution
│   ├── mcts.py
│   ├── minimax.py         # Depth-limited minimax, searches to round boundary
│   ├── move_filters.py    # non_floor_moves shared helper
│   ├── random.py          # Inherits uniform distribution
│   └── registry.py        # Single source of truth for all agents
├── api/
│   ├── main.py
│   └── schemas.py
├── engine/
│   ├── board.py           # Board dataclass + clone()
│   ├── constants.py       # Tile enum, WALL_PATTERN, FLOOR_PENALTIES, all constants
│   ├── game_recorder.py   # GameRecorder, GameRecord, RoundRecord, MoveRecord
│   ├── game_state.py      # GameState dataclass + clone()
│   ├── game.py            # Game controller: make_move, advance, legal_moves,
│   │                      #   score_round, count_distinct_source_color_pairs, etc.
│   ├── replay.py          # replay_to_move
│   └── scoring.py         # Pure scoring functions (incl. earned_score_unclamped)
├── frontend/
│   ├── game.js
│   ├── index.html
│   ├── render.js
│   └── style.css
├── neural/
│   ├── encoder.py         # (14,5,6) spatial + (49,) flat encoding
│   ├── model.py           # Conv+MLP trunk with 3 value heads (win/diff/abs)
│   ├── zobrist.py         # Zobrist hashing for within-round game states
│   ├── search_tree.py     # SearchTree: MCTS, transposition table, subtree reuse
│   ├── trainer.py         # compute_loss, Trainer, target functions, data collection
│   └── replay.py          # Circular buffer, three value targets per example
├── scripts/
│   ├── bench_score_placement.py
│   ├── benchmark_agents.py  # Time agents at various depth configs (first-move isolated)
│   ├── benchmark_mcts.py
│   ├── migrate_recordings.py
│   ├── parse_log.py
│   ├── self_play.py
│   ├── tournament.py        # Round-robin parallel tournament with per-agent timing
│   └── train.py
├── tests/                 # pytest suite (~649 tests; timing-sensitive ones marked slow)
├── checkpoints/           # gitignored
├── cli/                   # just enough to debug
├── htmlcov/               # gitignored
├── recordings/            # gitignored
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
### Phase 6 — AlphaZero Self-Play Training ✅ (superseded by 8c)
### Phase 6b — Reward Shaping + UI Polish ✅
### Phase 7 — Undo + Hypothetical + Manual Factory Setup ✅

---

### Phase 8 — Evaluation and Iteration 🔄 (in progress)

#### 8a — Search and Encoding Rewrite ✅
Spatial+flat encoding, conv+MLP model, Zobrist hashing, game-owned SearchTree with transposition table and subtree reuse, factory canonicalization, round boundaries as leaves.

#### 8b — Batched Multithreaded MCTS ✅
Virtual loss, parallel leaf collection, single batched forward pass per batch.

#### 8c — Heuristic Baseline ✅
Strong heuristic agents established. AlphaBeta hard (depths=3,5,8 thresholds=20,10) wins 99% vs Greedy at depth 1 alone, 76% vs Minimax. Inspector UI complete with immediate/cumulative scores, start/stop, copy state/tree. AlphaBeta is now both the pretrain opponent and the promotion bar for AlphaZero.

#### 8d — Encoding Upgrade ✅

**Changes shipped:**

- **Spatial shape: (12,5,6) → (14,5,6).** Two new channels added — one per player.
- **`blocked_wall` channel (ch 6 / 13).** A wall cell is marked 1.0 if it is already filled OR if the pattern line for that row is committed to a different color (so that cell cannot be filled this round). Previously the net had to learn this inference itself from wall + pattern line data. Now it is explicit.
- **`earned_score_unclamped` replaces `earned_score` in flat features (offsets 34–35).** The unclamped value correctly reflects floor penalties dragging scores negative and pending placement points not yet on board.score. The clamped version was hiding meaningful signal.
- **Score delta divisor: 20 → 50.** AlphaBeta vs weak agents produces large differentials; the wider range keeps the feature informative rather than saturated.
- **Round progress feature added (offset 47).** `(round - 1) / 5` gives 0.0 at round 1, 1.0 at round 6. Previously derivable only from bag counts — noisy and indirect.
- **Distinct source-color pairs feature added (offset 48).** `count_distinct_source_color_pairs() / 10`. Counts unique (source, color) combinations with tiles available across all factories and center (excluding FIRST_PLAYER). This is the maximum turns remaining in the round — a clean countdown. Also exposed as `Game.count_distinct_source_color_pairs()` for future use by AlphaBeta/Minimax depth selection.
- **Flat size: 47 → 49.**
- **Model `in_channels`: 12 → 14** (via `NUM_CHANNELS` import — no direct model.py edit needed).

**What was considered and rejected:**

- `pending_wall_placement` channel (binary: where will a committed pattern line land on the wall). Rejected — the existing color planes already encode this implicitly; the blocked_wall channel is the more valuable complement.
- `wall_nearly_complete` channel (bonus proximity). Rejected — the net can learn bonus proximity from wall state alone; adding prior round board state snapshots is a higher-value use of encoding complexity.
- `clamped_points` flat feature. Rejected — floor fill ratio already captures this; game-history artifacts are not actionable by the net.
- `value_abs` weight reduction to 0.0. Deferred — set to 0.1 for now, remove entirely after confirming it adds no signal.

**Deferred to Phase 8f:**
- Prior round board state snapshots (one encoded board state per completed round, up to 5). This would help the net learn that early-round center column plays create adjacency opportunities in later rounds, and provide a strong game-progress signal. Staged after a training run confirms the Phase 8d encoding improvements work.

#### 8e — Training Run 🔄 (next)

**Pretrain configuration:**
- Opponent: AlphaBeta easy (depths=2,3,7 thresholds=20,10) vs AlphaBeta medium (depths=3,5,7) — richer training data than Greedy-vs-Cautious, genuine wall-building structure, soft policy targets (uniform inherited distribution).
- `value_abs` weight: 0.1 (down from 0.3 — low confidence this head adds signal).
- `--value-only-iterations 0` always.
- `--clear-buffer-after-pretrain` never.

**Promotion bar:** Beat `alphabeta_hard` (depths=3,5,8 thresholds=20,10) at ≥55% win rate with ≥1500 eval simulations.

**Graduated eval targets** (checkpoints are promoted through these in order):
1. Beat Greedy (≥70%) — proves the net learned something real
2. Beat Cautious (≥60%) — confirms floor avoidance
3. Beat AlphaBeta easy (≥55%)
4. Beat AlphaBeta hard (≥55%) — deployable

**Smoke test command:**
```
python -m scripts.train \
  --iterations 3 \
  --games-per-iter 5 \
  --simulations 200 \
  --train-steps 200 \
  --pretrain-games 20 \
  --pretrain-steps 300 \
  --value-only-iterations 0 \
  --skip-eval-iterations 3 \
  --eval-games 10 \
  --eval-simulations 200 \
  --win-threshold 0.55
```

**Medium run:**
```
python -m scripts.train \
  --iterations 10 \
  --games-per-iter 15 \
  --simulations 750 \
  --train-steps 300 \
  --pretrain-games 100 \
  --pretrain-steps 1500 \
  --value-only-iterations 0 \
  --skip-eval-iterations 3 \
  --eval-games 20 \
  --eval-simulations 1500 \
  --win-threshold 0.55
```

**Before running:** trainer.py must be updated to use AlphaBeta (easy/medium) as the pretrain opponent pair instead of Greedy-vs-Cautious. This is the next coding task.

---

### Hard-won lessons (do not repeat)

1. **`value_only_iterations > 0` is a divergence trap.** Value head learns to accurately predict garbage outcomes while policy stays random. Self-play gets progressively worse. Always `--value-only-iterations 0`.

2. **One-hot policy targets from heuristic agents poison the policy head.** The policy head memorizes specific choices rather than learning structure. Fix: `policy_distribution()` on each agent returns its true (soft) sample distribution. Heuristic pretrain pushes these distributions as targets. Uniform inherited distribution (RandomAgent, MinimaxAgent, AlphaBetaAgent) is soft enough — it does NOT produce one-hots because legal moves number in the dozens.

3. **Random agent pretrain is nearly useless.** Random plays too many floor moves, producing near-zero `earned_score_unclamped` deltas across games. Value head gets no signal. Greedy-vs-Random was an improvement over random-only but still had one-hot policy target problems. Current recommendation: AlphaBeta easy vs AlphaBeta medium — genuine wall-building, meaningful score variance, soft distributions.

4. **Clearing the buffer after pretrain kills value head signal.** Early self-play data alone has near-uniform scores. Mixed buffer keeps pretrain signal alive during early iterations. Default: never clear.

5. **`_MAX_MOVES = 100` is the right cap.** Human games max at ~65 moves; anything longer is pathological and wastes compute.

6. **Eval at low simulation counts (100-200) is nearly useless.** With separate trees per agent, search quality is too thin. Win rates are ~50% + noise. Need ≥1500 sims for meaningful eval.

7. **GPU utilization is ~1%. CPU is the bottleneck.** MCTS inference runs on CPU deliberately. Parallelism via multiprocessing (not threads — GIL) gives near-linear speedup up to core count.

8. **`earned_score_unclamped` is the correct scoring primitive.** The clamped version hides floor penalties and pending placements. Use unclamped everywhere in the encoding and as the basis for value targets. Official `board.score` is only appropriate for win/loss determination.

9. **AlphaBeta is strictly superior to Minimax given equal depth.** AlphaBeta prunes without affecting result. Minimax is retained only because it produces a full search tree useful for human game analysis (no pruned branches). Do not use Minimax for training or eval.

10. **`count_distinct_source_color_pairs()` is a better depth-selection signal than `len(legal_moves())`.** Legal move count is O(factories × colors × rows) and spikes at round start. Distinct pairs count is O(factories × colors) and cleanly represents maximum remaining turns. Noted for future AlphaBeta/Minimax depth selection refactor.

---

### Engine design: make_move / advance separation (2026-04-20)

`make_move` now only moves tiles. All phase transitions are the caller's responsibility via `advance()`.

```
make_move(move)           — take tiles from source, place on pattern line/floor
advance(skip_setup=False) — next_player(), score_round() if round over,
                            score_game() if game over, setup_round() unless
                            skip_setup=True. Returns True if round boundary crossed.
is_round_over() → bool    — True when no color tiles remain in any source
next_player()             — rotate current_player (public, used by search tree)
score_round()             — wall scoring, floor penalties, set next first player
is_game_over() → bool     — True if any player has a completed wall row
score_game()              — end-of-game bonuses
setup_round()             — fill factories, add FIRST_PLAYER to center
count_distinct_source_color_pairs() → int
                          — unique (source, color) pairs with tiles available;
                            maximum turns remaining this round
```

---

### Multi-head value network

`AzulNet.forward(spatial, flat)` returns `(logits, value_win, value_diff, value_abs)`.

- `value_win` — win/loss outcome (+1/0/-1). Primary target. Only head used by PUCT during search.
- `value_diff` — normalized score differential (÷50). Auxiliary, dense gradient signal.
- `value_abs` — normalized absolute player score (÷100). Auxiliary weight reduced to 0.1; candidate for removal after next training run.

Loss: `policy + value_win + 0.3·value_diff + 0.1·value_abs`.

When adding callsites that consume net output:
```python
logits, value_win, _value_diff, _value_abs = net(spatial, flat)
```

---

### Policy distribution system

All agents have `policy_distribution(game) -> list[tuple[Move, float]]`.

- `Agent` base class: uniform over legal moves (default)
- `RandomAgent`: inherits default
- `CautiousAgent`: uniform over non-floor moves
- `EfficientAgent`: uniform over partial-line moves (fallback to all)
- `GreedyAgent`: color-conditional — pick color uniformly, then uniform within color
- `MinimaxAgent` / `AlphaBetaAgent`: uniform (inherited)

Used in `collect_heuristic_games` to produce soft policy targets. Uniform over dozens of moves is soft enough — not one-hot.

---

### Minimax/AlphaBeta agent notes

Both agents search to the round boundary naturally — depth limit is rarely reached. The adaptive depth system via `depths/thresholds` tuples controls depth based on legal move count:

```python
AlphaBetaAgent(depths=(3, 5, 8), thresholds=(20, 10))
# >20 legal moves -> depth 3 (early round, high branching)
# 10-20 legal moves -> depth 5
# <=10 legal moves  -> depth 8
```

**Future:** switch depth selection from `len(legal_moves())` to `count_distinct_source_color_pairs()` for a cleaner branching-factor signal.

Key invariant: **compute `earned_score_unclamped` BEFORE calling `advance(skip_setup=True)`**. After advance, the wall is scored and pattern lines cleared.

AlphaBeta move ordering uses cheap heuristic (no cloning):
- Floor moves: bad for maximizer, good for minimizer
- Line-completing moves: good for maximizer, bad for minimizer
- Partial fills: neutral

Clone-based `_immediate_score` ordering was 16x slower. Never reintroduce.

---

### Inspector UI (2026-04-22)

- Start/Pause toggle, sim count from `root.visits`
- Fully-explored detection — PUCT skips fully-explored nodes
- Per-move immediate score delta via `earned_score_unclamped`
- Cumulative minimax rollup of immediate scores along best line
- Children sorted by cumulative score, alternating desc/asc by depth
- Copy state / Copy tree buttons

---

### Agent registry

`agents/registry.py` is the single source of truth. `GET /agents` serves visible agents. Adding a new agent: registry entry + `PlayerType` in `schemas.py`.

UI difficulty levels:
- `alphabeta_easy`: `depths=(2,3,7), thresholds=(20,10)` — ~4ms/move
- `alphabeta_medium`: `depths=(3,5,7), thresholds=(20,10)` — ~35ms/move
- `alphabeta_hard`: `depths=(3,5,8), thresholds=(20,10)` — ~35ms/move, promotion bar

---

### Open issues

- **trainer.py pretrain opponent must be updated** to AlphaBeta easy vs medium before the next training run.
- **`value_abs` is a candidate for removal.** Weight reduced to 0.1. Remove entirely if next run shows no benefit.
- **Distinct pair count not yet used for AlphaBeta depth selection.** Noted for future refactor.
- **Prior round board state snapshots deferred.** High value for multi-round reasoning but requires architecture change. After 8e confirms basic training works.

---

### Deferred

- **Parallel self-play via multiprocessing.** CPU headroom confirmed. 2-4x speedup likely. File until training loop is stable.
- **Encoding cache keyed by Zobrist hash.** Saves ~19% of search time.
- **Shared state tree for two-agent eval.** Solves two-tree eval architecturally.
- **Prior round board state snapshots (Phase 8f).** One encoded board state per completed round, up to 5 rounds. Would help the net learn center-column adjacency value and game progress. Stage after 8e.
- **Inspector agent selector** — choose minimax/alphabeta as inspector backend.
- **Elo ladder** across all agent versions.
- **AlphaBeta depth selection via `count_distinct_source_color_pairs()`** instead of legal move count.

---

### Next up

- [ ] Update `trainer.py` pretrain to use AlphaBeta easy vs AlphaBeta medium
- [ ] Smoke test training run
- [ ] Medium training run
- [ ] Wire best checkpoint into API
- [ ] Add AlphaZero as UI opponent option

---

### Phase 9 — Polish and Release
- [ ] Animated tile placement
- [ ] Sound effects
- [ ] Cloud deployment
- [ ] Capacitor iOS/Android packaging
- [ ] README with screenshots

---

## Agent Hierarchy

| Agent | Strength | `policy_distribution` | Purpose |
|---|---|---|---|
| `RandomAgent` | Baseline | Uniform over legal (inherited) | Benchmark floor — avoid for training |
| `EfficientAgent` | ~22% overall | Uniform over partial-line | Weak — too passive |
| `CautiousAgent` | ~47% vs Greedy | Uniform over non-floor | Avoids penalties |
| `GreedyAgent` | ~49% overall | Color-conditional | No longer recommended as pretrain opponent |
| `MCTSAgent` | Untested vs new agents | (N/A) | Lookahead without neural net |
| `MinimaxAgent` | >> Greedy (100%) | Uniform (inherited) | Full tree for analysis only — not for training |
| `AlphaBetaAgent` | >> Minimax (76%) | Uniform (inherited) | Pretrain opponent + UI bot + promotion bar |
| `AlphaZeroAgent` | Goal: >> AlphaBeta hard | (via SearchTree) | Final goal |

---

## Key Principles

**TDD always.** Engine independence. Commit often. CI is the source of truth.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/train.py` | AlphaZero self-play training loop |
| `scripts/tournament.py` | Round-robin parallel tournament with per-agent timing and depth overrides |
| `scripts/benchmark_agents.py` | First-move vs overall timing by depth config |
| `scripts/self_play.py` | Generate self-play games |
| `scripts/parse_log.py` | Parse training logs |
| `scripts/migrate_recordings.py` | Migrate old recording format |

Tournament usage:
```
python -m scripts.tournament --agents greedy minimax alphabeta_hard --games 200 --workers 8
python -m scripts.tournament --agents minimax alphabeta --games 100 --workers 8 --depths0 2 3 5 --depths1 3 5 8 --thresholds1 20 10
```

Benchmark usage:
```
python -m scripts.benchmark_agents --games 3
```

---

## Change Log

| Date | Change |
|---|---|
| 2026-03-29 | Initial plan |
| 2026-04-01 | Phases 1-4 complete |
| 2026-04-02 | Phase 5 complete; Phase 6 in progress |
| 2026-04-03 | Phase 6 run 4 complete — failure analysis, Phase 6b defined |
| 2026-04-07 | Phase 6b complete; Phase 7 defined |
| 2026-04-13 | Phase 7 complete |
| 2026-04-14 | Phase 8a complete — spatial encoder, conv+MLP model, Zobrist hashing, SearchTree |
| 2026-04-14 | Phase 8b complete — batched multithreaded MCTS with virtual loss |
| 2026-04-15 | Phase 8c begun — eval fixes, move cap, loss diagnostic |
| 2026-04-18 | First long run diverged (value-only pathology). Applied lessons: full policy+value from iter 1, `_MAX_MOVES = 100`. |
| 2026-04-18 | Multi-head value network shipped (value_win / value_diff / value_abs with weighted loss). |
| 2026-04-18 | Distributional policy targets shipped — `policy_distribution()` on all agents. |
| 2026-04-18 | Eval two-tree problem identified. |
| 2026-04-20 | Engine refactor: make_move decoupled from round/game transitions. advance() owns phase loop. |
| 2026-04-20 | API fix: advance() called unconditionally after make_move. |
| 2026-04-22 | Phase 8c complete. Inspector UI, SearchTree fixes, MinimaxAgent, AlphaBetaAgent, agent registry, tournament script, earned_score_unclamped. |
| 2026-04-22 | Phase 8d complete. Encoding upgrade: blocked_wall channel (14 channels), earned_score_unclamped in flat features, score delta divisor 50, round progress, distinct source-color pairs. Game.count_distinct_source_color_pairs() added. value_abs weight reduced to 0.1. Pretrain switched to AlphaBeta easy vs medium. 649 tests passing. |