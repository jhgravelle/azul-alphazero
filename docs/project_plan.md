# Azul AlphaZero — Project Plan

> Last updated: 2026-04-18 (evening)
> Status: Phase 8c in progress. Multi-head value network shipped. Distributional policy targets working. Eval two-tree dynamics identified as the current bottleneck for checkpoint promotion.

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

**Compute split**: MCTS inference runs on CPU (faster for small batch sizes and this model), training runs on GPU. This means GPU sits near-idle during self-play and eval; CPU is the actual bottleneck.

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
│   ├── base.py            # Agent base class + default policy_distribution (uniform)
│   ├── random.py          # Inherits uniform distribution
│   ├── cautious.py        # Uniform over non-floor moves
│   ├── efficient.py       # Uniform over partial-line moves (fallback to all)
│   ├── greedy.py          # Color-conditional distribution
│   ├── mcts.py
│   ├── move_filters.py    # non_floor_moves shared helper
│   └── alphazero.py       # Thin wrapper — delegates to SearchTree
├── neural/
│   ├── encoder.py         # (12,5,6) spatial + (47,) flat encoding
│   ├── model.py           # Conv+MLP trunk with 3 value heads (win/diff/abs)
│   ├── zobrist.py         # Zobrist hashing for within-round game states
│   ├── search_tree.py     # SearchTree: MCTS, transposition table, subtree reuse
│   ├── trainer.py         # compute_loss, Trainer, target functions, data collection
│   └── replay.py          # Circular buffer, three value targets per example
├── api/
│   ├── main.py
│   └── schemas.py
├── frontend/
├── scripts/
│   ├── self_play.py
│   ├── train.py
│   └── migrate_recordings.py
├── tests/                 # pytest suite (~540 tests; timing-sensitive ones marked slow)
├── recordings/            # gitignored
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
### Phase 6 — AlphaZero Self-Play Training ✅ (superseded by 8c)
### Phase 6b — Reward Shaping + UI Polish ✅
### Phase 7 — Undo + Hypothetical + Manual Factory Setup ✅

---

### Phase 8 — Evaluation and Iteration 🔄 (in progress)

#### 8a — Search and Encoding Rewrite ✅
Spatial+flat encoding, conv+MLP model, Zobrist hashing, game-owned SearchTree with transposition table and subtree reuse, factory canonicalization, round boundaries as leaves.

#### 8b — Batched Multithreaded MCTS ✅
Virtual loss, parallel leaf collection, single batched forward pass per batch.

#### 8c — Training Run + Iteration 🔄 (in progress)

**Hard-won lessons (do not repeat)**

1. **`value_only_iterations=15` is a divergence trap.** Value head learns to accurately predict garbage outcomes while policy stays random. Self-play gets progressively worse. Always train policy+value together from iter 1.

2. **One-hot policy targets from heuristic agents poison the policy head.** The policy head memorizes Greedy's specific choices rather than learning structure. Fix: `policy_distribution()` on each agent returns its true (soft) sample distribution; heuristic games push these distributions as targets.

3. **Pretraining on Greedy-vs-Random gives wide-variance value data but one-hot policy targets.** Pretraining on Greedy-vs-Cautious gives narrow-variance value data but cleaner policy targets. Current choice: Cautious. Trade-off: `value_abs` starts with weaker signal because Cautious games cluster in a narrow score range.

4. **Clearing the buffer after pretrain killed `value_abs`.** With only self-play data (which has uniform-bad scores early), the absolute-score target has near-zero variance and the head learns a near-constant. Mixed buffer keeps the head alive. Default is now to *not* clear.

5. **`_MAX_MOVES = 100` is the right cap.** Pathological games get truncated; real games finish well under it.

6. **Eval at low simulation counts (100-200) is nearly useless.** With two separate trees (one per agent), search quality is so thin that eval games hit the move cap constantly and win rates are ~50% + noise. Needs either much deeper search (1500+ sims) or a shared-state tree design.

7. **GPU utilization is ~1%. CPU is the bottleneck.** The model is small and runs on CPU for MCTS inference. Parallelism via multiprocessing (not threads, because GIL) should give near-linear speedup up to core count.

**Multi-head value network**

`AzulNet.forward(spatial, flat)` returns `(logits, value_win, value_diff, value_abs)`.

- `value_win` — win/loss outcome (+1/0/-1). Primary target. Only head used by PUCT during search.
- `value_diff` — normalized score differential (`±20 → ±1.0`). Auxiliary, dense gradient.
- `value_abs` — normalized absolute player score (`50 → 1.0`). Auxiliary, teaches "score positive."

Loss: `policy + value_win + 0.3·value_diff + 0.3·value_abs`.

`ReplayBuffer` stores three value targets per example. `compute_loss` returns per-head breakdown for logging. All data collection sites compute all three targets from final scores.

**Policy distribution system**

`Agent` base class has `policy_distribution(game) -> list[tuple[Move, float]]` returning the distribution each agent samples from. Default: uniform over legal moves (matches `RandomAgent`). Overridden in Cautious, Efficient, Greedy.

`collect_heuristic_games` uses Greedy vs Cautious and pushes each agent's distribution as the policy target instead of one-hot.

**Training behavior observed (2026-04-18)**

First run with all fixes in place:
- Self-play scores improve iter-over-iter: -50 → -40 → -34 → -31 → -33 (200 sims)
- Policy loss moves as expected (natural target entropy floor around 3.7-3.8)
- Eval at 200 sims: all games hit 100-move cap, win rates ~50% + noise (useless)
- 750-sim self-play / 1500-sim eval experiment in progress to confirm eval with more search produces real measurement

**Open issues**

- **Eval win rate is a noisy metric at low sims.** Under test: does 1500-sim eval give meaningful measurement?
- **`value_abs` MSE is very low (0.03-0.15).** Could mean well-trained; could mean collapsed to near-constant. Investigate if training plateaus.
- **Eval move cap warning off-by-one** — logs `eval game N` for zero-indexed `i`. Cosmetic.

**Deferred / investigated but not prioritized**

- **Parallel self-play via multiprocessing.** CPU headroom confirmed (GPU at 1%). 2-4x speedup likely. ~1 day of work with TDD. File until training loop is stable.
- **`fully_explored` flag on SearchTree nodes.** 5-10% speedup at high sim counts on late-round subtrees.
- **Encoding cache keyed by Zobrist hash.** Saves ~19% of search time.
- **Shared state tree for two-agent eval.** Solves two-tree eval architecturally.
- **Time-varying auxiliary weights.** Curriculum — start with high `value_abs` weight, taper.
- **`NoSelfHarmAgent` / richer heuristic agents** to diversify pretraining data.
- **Mixing Greedy-vs-Random games back in** for `value_abs` variance if it goes dead.

**Next up**

- [ ] Finish 750/1500 sim experiment; confirm whether eval at 1500 sims produces meaningful measurement
- [ ] If yes, scale to a longer run with promotion working
- [ ] Wire best checkpoint into API `_make_agent()`
- [ ] Add AlphaZero as UI opponent option
- [ ] Elo ladder across all agent versions
- [ ] Difficulty levels in UI

---

### Phase 9 — Polish and Release
- [ ] Animated tile placement
- [ ] Sound effects
- [ ] Cloud deployment
- [ ] Capacitor iOS/Android packaging
- [ ] README with screenshots

---

## Agent Hierarchy

| Agent | Heuristics | `policy_distribution` | Purpose |
|---|---|---|---|
| `RandomAgent` | None | Uniform over legal (inherited) | Benchmark baseline |
| `CautiousAgent` | Floor-avoidance | Uniform over non-floor | Avoids penalties |
| `EfficientAgent` | Partial-line preference | Uniform over partial-line | Completes lines faster |
| `GreedyAgent` | Both heuristics | Color-conditional | Default UI opponent |
| `MCTSAgent` | UCB1 + random rollouts | (N/A) | Lookahead without neural net |
| `AlphaZeroAgent` | PUCT + neural net | (via SearchTree) | Final goal |

---

## Key Principles

**TDD always.** Engine independence. Commit often. CI is the source of truth.

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
| 2026-04-18 | First long run diverged (value-only pathology). Applied lessons: full policy+value training from iter 1, `_MAX_MOVES = 100`. |
| 2026-04-18 | Multi-head value network shipped (value_win / value_diff / value_abs with weighted loss). `--clear-buffer-after-pretrain` flag added. |
| 2026-04-18 | Distributional policy targets shipped — `policy_distribution()` on all agents, heuristic pretrain switched to Greedy vs Cautious. First training run with observable self-play improvement. |
| 2026-04-18 | Eval two-tree problem identified as main blocker for checkpoint promotion. GPU at 1% — CPU is the bottleneck; parallelism deferred pending stable training loop. |