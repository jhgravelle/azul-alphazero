# Azul AlphaZero — Project Plan

> Last updated: 2026-04-24
> Status: Phase 8f in progress. Encoder v2 complete — 8-channel (8,5,5) spatial, 8-value flat, 56% input size reduction. Factory fingerprinting eliminated structurally. Phase 1 diff-only training run in progress (30 iterations, AlphaBeta heuristic games, no self-play). Next: diagnose value_diff calibration via sample_policy, then Phase 2 full training.

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
│   ├── alphabeta.py       # Alpha-beta pruning, scored policy_distribution (softmax)
│   ├── base.py            # Agent base class + default policy_distribution (uniform)
│   ├── cautious.py        # Uniform over non-floor moves
│   ├── efficient.py       # Uniform over partial-line moves (fallback to all)
│   ├── greedy.py          # Color-conditional distribution
│   ├── mcts.py
│   ├── minimax.py         # Depth-limited minimax, full tree (no pruning) for analysis
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
│   ├── encoder.py         # (8,5,5) spatial + (8,) flat encoding (rewritten 2026-04-24)
│   ├── model.py           # Conv+MLP trunk with 3 value heads (win/diff/abs)
│   ├── zobrist.py         # Zobrist hashing for within-round game states
│   ├── search_tree.py     # SearchTree: MCTS, transposition table, subtree reuse
│   ├── trainer.py         # compute_loss, Trainer, target functions, data collection
│   └── replay.py          # Circular buffer, three value targets per example
├── scripts/
│   ├── bench_score_placement.py
│   ├── benchmark_agents.py
│   ├── benchmark_mcts.py
│   ├── inspect_policy.py  # Per-move policy/value/MCTS diagnostic tool
│   ├── sample_policy.py   # Bulk value head calibration checker (N random states)
│   ├── migrate_recordings.py
│   ├── parse_log.py
│   ├── self_play.py
│   ├── tournament.py
│   └── train.py
├── tests/
├── checkpoints/           # gitignored; latest.pt always = most recent training state
├── cli/
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
### Phase 6 — AlphaZero Self-Play Training ✅ (superseded)
### Phase 6b — Reward Shaping + UI Polish ✅
### Phase 7 — Undo + Hypothetical + Manual Factory Setup ✅

---

### Phase 8 — Evaluation and Iteration 🔄 (in progress)

#### 8a — Search and Encoding Rewrite ✅
Spatial+flat encoding, conv+MLP model, Zobrist hashing, game-owned SearchTree with transposition table and subtree reuse, factory canonicalization, round boundaries as leaves.

#### 8b — Batched Multithreaded MCTS ✅
Virtual loss, parallel leaf collection, single batched forward pass per batch.

#### 8c — Heuristic Baseline ✅
AlphaBeta hard wins 99% vs Greedy at depth 1, 76% vs Minimax. Inspector UI complete. AlphaBeta is both pretrain opponent and promotion bar.

#### 8d — Encoding Upgrade ✅

Spatial: (12,5,6) → (14,5,6). Two new channels per player:
- **`blocked_wall`** (ch 6/13): wall cell is 1.0 if already filled OR if pattern line committed to a different color. Makes "this cell is unavailable" explicit rather than inferred.

Flat: 47 → 49 features:
- `earned_score_unclamped` replaces `earned_score` at offsets 34-35
- Score delta divisor: 20 → 50 (AlphaBeta games have larger spreads)
- Round progress: `(round-1)/5` at offset 47
- Distinct source-color pairs / 10 at offset 48 (round countdown proxy)

`Game.count_distinct_source_color_pairs()` added — cleaner depth-selection signal than legal move count.

#### 8e — Training Pipeline + Diagnostics ✅

**Major infrastructure shipped:**

- **AlphaBeta scored policy distribution.** `policy_distribution()` returns softmax over root move scores (temperature=1.0) rather than uniform. `_score_all_root_moves` evaluates every root move with full alpha/beta window (no root-level pruning). `choose_move` must be called before `policy_distribution` — populates `_root_move_scores` cache. Falls back to uniform if cache empty.

- **Diverse heuristic matchups.** `collect_heuristic_games` uses weighted matchup sampling: Random/Efficient/Cautious/Greedy/AlphaBeta easy vs AlphaBeta medium, plus medium vs medium. Gives value head spectrum of position quality including losing positions from weak agents.

- **Parallel heuristic collection.** `collect_heuristic_games_parallel` uses `multiprocessing` with `spawn` context (Windows compatible). Workers serialize agent names not callables (callables don't pickle). Near-linear speedup up to core count.

- **Training loop fixes.** Net weights no longer reset when eval is skipped (critical bug). Interval loss logging fixed (was cumulative, now per-500-step window). `--diff-only` flag for value-only-differential training. `--initial-generation` for manually promoted checkpoints. `latest.pt` auto-loaded by default; written every iteration and on promotion.

- **Diagnostic scripts.** `inspect_policy.py`: per-move AlphaBeta vs net policy, KL divergence, value heads, encoding verification, MCTS probe with top-child subtree analysis. `sample_policy.py`: bulk value calibration across N random states, mean/std/min/max for all heads, floor preference rate.

#### 8f — Encoder v2 🔄 (in progress)

Complete redesign of the encoder to eliminate factory fingerprinting and reduce input size.

**Spatial: (14,5,6) → (8,5,5).** All channels now in (row, wall_col) space — convolutions operate on true geometric adjacency. 56% reduction in input values (469 → 208).

Channel layout:
- **ch 0/3** — My/opponent wall filled (0 or 1). Conv learns adjacency implicitly.
- **ch 1/4** — My/opponent pattern line fill ratio. Nonzero at exactly one wall_col per row (the committed color's column). 0 if line empty or wall cell already filled.
- **ch 2/5** — My/opponent bonus proximity. Weighted sum of row/col/color completion progress, using pattern tile cost weighting `(row+1)` per filled cell, max 15. Includes partial pattern line contributions. Formula: `((15 - weighted_row) + (15 - weighted_col) + (15 - weighted_color)) / (3 * 15)`. Goes negative for expensive bottom rows at game start — intentional.
- **ch 6** — Bag count by color, broadcast across rows. col = COLOR_TILES index. value = count / 20.
- **ch 7** — Source distribution. row = bucket (sources with 1/2/3/4/5+ tiles), col = color. value = source count / 5. Eliminates factory fingerprinting — encodes tile availability and distribution without identifying which factory has which tiles.

**Flat: (49,) → (8,).** Dropped: per-factory counts (memorization culprit), center counts (rolled into ch 7), score delta (derivable), discard (marginal signal), round progress (bag counts sufficient), distinct pairs (ch 7 covers this). Kept: official scores, earned-this-round unclamped, floor penalty (actual penalty value / 14, not fill ratio), first-player token flags.

**All existing tests passing.** Old checkpoints incompatible — deleted. Fresh training required.

---

### Hard-won training lessons (do not repeat)

**From earlier phases:**

1. **`value_only_iterations > 0` is a divergence trap.** Policy stays random, value predicts garbage. Always `--value-only-iterations 0`.

2. **One-hot policy targets poison the policy head.** Fix: `policy_distribution()` returns soft distributions. AlphaBeta now returns softmax over search scores.

3. **Random agent pretrain is nearly useless.** Floor-heavy games, near-zero score signal. Use structured opponents.

4. **Clearing buffer after pretrain kills value signal.** Never use `--clear-buffer-after-pretrain`.

5. **`_MAX_MOVES = 100` is the right game cap.**

6. **Eval at low sim counts is nearly useless.** Need ≥1500 sims for meaningful eval with separate trees.

**From Phase 8e (2026-04-23):**

7. **Net weights reset every iteration when eval was skipped.** Non-promoted branch called `net.load_state_dict(best_net)` unconditionally. With `--skip-eval-iterations 20`, every iteration reset to random weights — 9 iterations of training discarded silently each run. Fix: only reset when eval actually ran and net lost.

8. **Policy loss dominates the trunk (~50x value loss).** With policy loss ~2.0 and value loss ~0.04, trunk receives far more gradient from policy than value heads. During Phase 1, train value-only (`--diff-only`) so trunk develops value-relevant features before policy training dominates.

9. **Value head overfit to factory configurations.** Five fresh empty boards produced `value_diff` of [-0.998, +0.362, -0.453, +0.802, -0.297]. Factory tile configuration at move 1 has almost zero bearing on game outcome, but net memorized factory pattern → outcome correlations. Each factory draw is effectively unique — net sees it once and memorizes it rather than generalizing. **Fix: encoder v2 eliminates per-factory counts entirely, replacing with source distribution buckets.**

10. **Model capacity too large for available data.** 256-dim trunk with 3 residual blocks memorizes unique configurations rather than generalizing. Dropout deferred — encoder redesign is the structural fix; revisit if std > 0.3 persists after encoder v2 training.

11. **AlphaBeta policy distribution is flat at move 1 on empty boards (depth 1).** Most moves score 0 at depth 1 — partial fills have no immediate scoring, factory overflows cancel with floor penalties. Score variation only appears at depth ≥3 or when pattern lines are partially filled. This is correct behavior. Policy head learns from mid/late-round positions where distributions are peaked.

12. **MCTS snowballs on high-value outliers.** Noisy value head gives one move a high estimate on first visit; PUCT concentrates all simulations there regardless of policy prior. Fix: value head calibration, not PUCT change.

13. **Eval against random `gen_0000.pt` produces move-cap hits.** Random net floors constantly; both players score -90 to -106 and hit 100-move cap. Fix: copy `latest.pt` to `gen_0001.pt` after Phase 1 and use `--initial-generation 1`.

14. **`uvicorn --reload` restarts server on checkpoint writes.** Fix: run without `--reload` during training.

**From Phase 8f (2026-04-24):**

15. **Factory fingerprinting is a structural encoder problem, not a capacity problem.** Per-factory per-color counts are a near-unique fingerprint for each game start. The fix is to never encode which factory has which tiles — only encode aggregate availability (source distribution buckets). Dropout addresses symptoms; encoder redesign addresses cause.

16. **Convolutions handle adjacency — don't encode it explicitly.** Explicit adjacency channels are redundant when conv layers operate on a true geometric grid. A 3×3 conv already computes weighted neighbor sums. Removing adjacency channels simplifies the encoder without losing information.

17. **Wall encoding space matters.** Previous encoder used (row, color) indexing — color planes with a diagonal structure the net had to reverse-engineer. Encoder v2 uses (row, wall_col) space where geometric adjacency in the tensor matches geometric adjacency on the board.

18. **Bonus proximity should use pattern tile cost weighting, not cell counts.** Filling row 4 costs 5 pattern tiles; row 0 costs 1. Weighting by `(row+1)` makes the progress signal reflect true tile efficiency. Max weighted sum per row/col/color = 15. Values go negative for expensive rows at game start — this is informative, not a bug.

---

### Engine design: make_move / advance separation

```
make_move(move)           — take tiles from source, place on pattern line/floor
advance(skip_setup=False) — next_player(), score_round() if round over,
                            score_game() if game over, setup_round() unless
                            skip_setup=True. Returns True if round boundary crossed.
is_round_over() → bool    — True when no color tiles remain in any source
next_player()             — rotate current_player
score_round()             — wall scoring, floor penalties, set next first player
is_game_over() → bool     — True if any player has a completed wall row
score_game()              — end-of-game bonuses
setup_round()             — fill factories, add FIRST_PLAYER to center
count_distinct_source_color_pairs() → int
                          — unique (source, color) pairs with tiles; max turns remaining
```

---

### Multi-head value network

`AzulNet.forward(spatial, flat)` returns `(logits, value_win, value_diff, value_abs)`.

- `value_win` — win/loss (+1/0/-1). Primary target.
- `value_diff` — normalized score differential (÷50). Dense continuous signal. **Currently used by PUCT** in `search_tree.py` (`make_policy_value_fn`).
- `value_abs` — normalized absolute score (÷100). Weight 0.1; candidate for removal.

Loss weights (full training): `policy + 0.3·value_win + 1.0·value_diff + 0.1·value_abs`
Loss weights (`--diff-only`): `value_diff` only — policy, value_win, value_abs all zeroed.

**Value head calibration target:** `value_diff` mean < |0.1|, std < 0.3 on turn-1 empty boards.
**Previous state:** mean ≈ -0.01 (good), std ≈ 0.64 (too high — factory memorization, now fixed structurally).

---

### Training pipeline reference

**Data sources per iteration (Phase 2):**

| Source | Flag | Purpose |
|---|---|---|
| Self-play (AZ vs AZ) | `--games-per-iter` | Net learns from own search |
| AlphaBeta vs AlphaBeta | `--alphabeta-games-per-iter` | High-quality imitation data |
| AlphaBeta vs candidate | `--candidate-games-per-iter` | Mixed signal, both sides soft policy |
| Eval games (AZ vs best) | `--eval-games` | Diversity; feeds buffer automatically |

**Heuristic matchup weights (default):**

| Matchup | Weight | Purpose |
|---|---|---|
| Random vs AlphaBeta medium | 10% | Extreme loss signal, fast games |
| Efficient vs AlphaBeta medium | 10% | Weak vs strong |
| Cautious vs AlphaBeta medium | 15% | Moderate loss signal |
| Greedy vs AlphaBeta medium | 20% | Near-peer, clean policy targets |
| AlphaBeta easy vs medium | 25% | Peer matchup |
| AlphaBeta medium vs medium | 20% | Symmetric high quality |

**Value calibration check (run after each phase):**
```
python -m scripts.sample_policy --checkpoint checkpoints/latest.pt --samples 100 --summary-only
```
Target: `value_diff` mean < |0.1|, std < 0.3 on turn-1 empty boards.

**Medium run — Phase 1 (value calibration, diff-only, encoder v2):**
```
python -m scripts.train \
  --iterations 30 --games-per-iter 0 --simulations 200 \
  --train-steps 1000 --value-only-iterations 0 \
  --skip-eval-iterations 30 --eval-games 0 --eval-simulations 200 \
  --win-threshold 0.55 --alphabeta-games-per-iter 100 \
  --buffer-size 500000 --heuristic-workers 8 --diff-only
```

**Medium run — Phase 2 (full training):**
```
python -m scripts.train \
  --iterations 30 --games-per-iter 10 --simulations 200 \
  --train-steps 1000 --value-only-iterations 0 \
  --skip-eval-iterations 5 --eval-games 20 --eval-simulations 200 \
  --win-threshold 0.55 --alphabeta-games-per-iter 40 \
  --candidate-games-per-iter 20 --buffer-size 200000 \
  --heuristic-workers 8 --initial-generation 1
```

**Promotion bar:** Beat `alphabeta_hard` at ≥55% win rate with ≥1500 eval simulations.

**Graduated eval targets:**
1. Beat Greedy ≥70%
2. Beat Cautious ≥60%
3. Beat AlphaBeta easy ≥55%
4. Beat AlphaBeta hard ≥55% — deployable

---

### Checkpoint management

- `checkpoints/latest.pt` — current training weights; written every iteration and on promotion
- `checkpoints/latest_params.json` — args used to produce `latest.pt`
- `checkpoints/gen_xxxx.pt` — promoted checkpoints only
- `--load` defaults to `checkpoints/latest.pt`; silently skips if file doesn't exist
- `--initial-generation N` — start generation counter at N

To manually promote Phase 1 checkpoint before Phase 2:
```
copy checkpoints\latest.pt checkpoints\gen_0001.pt
```
Then run Phase 2 with `--initial-generation 1`.

**Note:** Checkpoints are tied to encoder/model architecture. Encoder v2 checkpoints are incompatible with encoder v1. All v1 checkpoints deleted 2026-04-24.

---

### Inspector UI

- Start/Pause toggle, sim count from `root.visits`
- Fully-explored detection — PUCT skips fully-explored nodes
- Per-move immediate score delta via `earned_score_unclamped`
- Cumulative minimax rollup of immediate scores
- Children sorted by cumulative score, alternating desc/asc by depth
- Copy state / Copy tree buttons
- Run server without `--reload` during training

---

### Agent registry

`agents/registry.py` is the single source of truth. `GET /agents` serves visible agents. Adding a new agent: registry entry + `PlayerType` in `schemas.py`.

UI difficulty levels:
- `alphabeta_easy`: `depths=(2,3,7), thresholds=(20,10)` — ~4ms/move
- `alphabeta_medium`: `depths=(3,5,7), thresholds=(20,10)` — ~35ms/move
- `alphabeta_hard`: `depths=(3,5,8), thresholds=(20,10)` — ~35ms/move, promotion bar

---

### Open issues

- **value_diff std after encoder v2** — previous std ≈ 0.64 was due to factory fingerprinting. Run sample_policy after Phase 1 to confirm std < 0.3. If still high, consider dropout.
- **PUCT uses `value_diff` not `value_win`** — evaluate switching once value_diff is calibrated.
- **`_terminal_value` uses clamped `earned_score`** — mismatch with training targets. Low priority.
- **Move cap of 100 in eval** — too low when nets are poorly calibrated. Consider `--eval-move-cap 300`.

### Deferred

- **Model dropout** — deferred; encoder redesign is the structural fix for memorization. Revisit if std > 0.3 persists after Phase 1.
- **Prior round board state snapshots** — one encoded board state per completed round, up to 5.
- **AlphaBeta depth selection via `count_distinct_source_color_pairs()`** — cleaner than legal move count.
- **Parallel self-play via multiprocessing** — heuristic collection now parallel; self-play still sequential.
- **Encoding cache keyed by Zobrist hash** — saves ~19% search time.
- **Shared state tree for two-agent eval** — solves two-tree eval architecturally.
- **Inspector agent selector** — choose minimax/alphabeta as inspector backend.
- **Elo ladder** across all agent versions.

### Next up

- [ ] Phase 1 training run complete — check loss curve for learning vs overfit
- [ ] Run sample_policy — confirm value_diff std < 0.3 on empty boards
- [ ] If calibrated: copy latest.pt to gen_0001.pt, run Phase 2
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
| `GreedyAgent` | ~49% overall | Color-conditional | Training opponent (weak side) |
| `MCTSAgent` | Untested vs new agents | (N/A) | Lookahead without neural net |
| `MinimaxAgent` | >> Greedy (100%) | Uniform (inherited) | Full tree for analysis only |
| `AlphaBetaAgent` | >> Minimax (76%) | Softmax over search scores | Pretrain opponent + UI bot + promotion bar |
| `AlphaZeroAgent` | Goal: >> AlphaBeta hard | (via SearchTree MCTS visits) | Final goal |

---

## Key Principles

**TDD always.** Engine independence. Commit often. CI is the source of truth.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/train.py` | AlphaZero self-play training loop |
| `scripts/inspect_policy.py` | Per-move policy/value/MCTS diagnostic; encoding verification |
| `scripts/sample_policy.py` | Bulk value head calibration across N random game states |
| `scripts/tournament.py` | Round-robin parallel tournament with per-agent timing |
| `scripts/benchmark_agents.py` | First-move vs overall timing by depth config |
| `scripts/self_play.py` | Generate self-play games |
| `scripts/parse_log.py` | Parse training logs |
| `scripts/migrate_recordings.py` | Migrate old recording format |

Sample policy usage:
```
python -m scripts.sample_policy --checkpoint checkpoints/latest.pt --samples 100 --summary-only
python -m scripts.sample_policy --checkpoint checkpoints/latest.pt --samples 20 --turn 5
```

Inspect policy usage:
```
python -m scripts.inspect_policy --checkpoint checkpoints/latest.pt --moves 3 --depth 1 --mcts-sims 200
python -m scripts.inspect_policy --moves 2 --depth 1 --mcts-sims 50 --show-encoding
```

Tournament usage:
```
python -m scripts.tournament --agents greedy minimax alphabeta_hard --games 200 --workers 8
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
| 2026-04-18 | First long run diverged (value-only pathology). Applied lessons. |
| 2026-04-18 | Multi-head value network shipped. Distributional policy targets shipped. |
| 2026-04-20 | Engine refactor: make_move decoupled from round/game transitions. |
| 2026-04-22 | Phase 8c complete. Inspector UI, SearchTree fixes, MinimaxAgent, AlphaBetaAgent, registry, tournament, earned_score_unclamped. |
| 2026-04-22 | Phase 8d complete. Encoding upgrade: blocked_wall channel, unclamped scores, round progress, distinct pairs. |
| 2026-04-23 | Phase 8e complete. AlphaBeta scored policy distribution. Diverse heuristic matchups. Parallel heuristic collection. Training loop net-reset bug fixed. diff-only mode. Checkpoint management improvements. Diagnostic scripts: inspect_policy, sample_policy. Value head overfit to factory configurations identified (std=0.64, target <0.3). |
| 2026-04-24 | Phase 8f begun. Encoder v2: (8,5,5) spatial + (8,) flat, 56% input reduction. Wall-col space encoding, bonus proximity with pattern-tile-cost weighting and partial line contributions, source distribution buckets eliminate factory fingerprinting. All tests passing. Phase 1 diff-only training run started. |