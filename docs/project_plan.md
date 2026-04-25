# Azul AlphaZero — Project Plan

> Last updated: 2026-04-25
> Status: Phase 8e in progress. Inspector upgraded with agent selector (Minimax, AlphaBeta presets, AlphaZero), policy priors, visit fractions, and heuristic value head. Minimax inspector value fixed to use earned_score_unclamped ÷ 50. Double policy_value_fn call per node eliminated. Several inspector UI todos identified. Next: medium training run with model dropout + value calibration target (value_diff std < 0.3 on empty boards).

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
│   ├── encoder.py         # (14,5,6) spatial + (49,) flat encoding (rewritten 2026-04-23)
│   ├── model.py           # Conv+MLP trunk with 3 value heads (win/diff/abs) + dropout
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

**Encoder fully rewritten 2026-04-23** — all tests passing after rewrite.

#### 8e — Training Pipeline + Diagnostics 🔄 (in progress)

**Major infrastructure shipped:**

- **AlphaBeta scored policy distribution.** `policy_distribution()` returns softmax over root move scores (temperature=1.0) rather than uniform. `_score_all_root_moves` evaluates every root move with full alpha/beta window (no root-level pruning). `choose_move` must be called before `policy_distribution` — populates `_root_move_scores` cache. Falls back to uniform if cache empty.

- **Diverse heuristic matchups.** `collect_heuristic_games` uses weighted matchup sampling: Random/Efficient/Cautious/Greedy/AlphaBeta easy vs AlphaBeta medium, plus medium vs medium. Gives value head spectrum of position quality including losing positions from weak agents.

- **Parallel heuristic collection.** `collect_heuristic_games_parallel` uses `multiprocessing` with `spawn` context (Windows compatible). Workers serialize agent names not callables (callables don't pickle). Near-linear speedup up to core count.

- **Training loop fixes.** Net weights no longer reset when eval is skipped (critical bug). Interval loss logging fixed (was cumulative, now per-500-step window). `--diff-only` flag for value-only-differential training. `--initial-generation` for manually promoted checkpoints. `latest.pt` auto-loaded by default; written every iteration and on promotion.

- **Diagnostic scripts.** `inspect_policy.py`: per-move AlphaBeta vs net policy, KL divergence, value heads, encoding verification, MCTS probe with top-child subtree analysis. `sample_policy.py`: bulk value calibration across N random states, mean/std/min/max for all heads, floor preference rate.

- **Inspector agent selector.** Dropdown in inspector header selects backend: Minimax, AlphaBeta Easy/Medium/Hard/Extreme/Ludacris, AlphaZero. AlphaBeta shows preset parameters as read-only label. AlphaZero shows sim count dropdown (50/100/200/500/1000/5000). Changing agent or sims resets and restarts tree. Backend builds appropriate `policy_value_fn` per selection.

- **Inspector node stats.** Each node now shows: policy prior (`p=X.X%`), visit count, visit fraction of parent (`X%`), net value diff (raw net output before search). `visit_fraction` added to `_serialize_node`. `_ensure_expanded` now returns value to eliminate double `policy_value_fn` call per node.

- **Minimax inspector value head fixed.** `_make_minimax_pv` now returns `(current_score - opponent_score) / 50.0` as value (using `earned_score_unclamped`), giving MCTS a dense signal at every node rather than only at round boundaries. Divisor is 50 to match training targets. No clamp — Azul score differentials rarely exceed ±0.6 mid-round.

- **Copy Tree includes agent header.** Agent name and parameter string prepended to copied tree text.

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

9. **Value head overfit to factory configurations.** Five fresh empty boards produced `value_diff` of [-0.998, +0.362, -0.453, +0.802, -0.297]. Factory tile configuration at move 1 has almost zero bearing on game outcome, but net memorized factory pattern → outcome correlations. Each factory draw is effectively unique — net sees it once and memorizes it rather than generalizing. Fix: model dropout + smaller hidden_dim during Phase 1.

10. **Model capacity too large for available data.** 256-dim trunk with 3 residual blocks (~2M params) memorizes 500k unique factory configurations rather than generalizing. Fix: add dropout, optionally reduce hidden_dim.

11. **AlphaBeta policy distribution is flat at move 1 on empty boards (depth 1).** Most moves score 0 at depth 1 — partial fills have no immediate scoring, factory overflows cancel with floor penalties. Score variation only appears at depth ≥3 or when pattern lines are partially filled. This is correct behavior. Policy head learns from mid/late-round positions where distributions are peaked.

12. **MCTS snowballs on high-value outliers.** Noisy value head gives one move a high estimate on first visit; PUCT concentrates all simulations there regardless of policy prior. Fix: value head calibration, not PUCT change.

13. **Eval against random `gen_0000.pt` produces move-cap hits.** Random net floors constantly; both players score -90 to -106 and hit 100-move cap. Fix: copy `latest.pt` to `gen_0001.pt` after Phase 1 and use `--initial-generation 1`.

14. **`uvicorn --reload` restarts server on checkpoint writes.** Fix: run without `--reload` during training.

**From Phase 8e (2026-04-25):**

15. **`_ensure_expanded` called policy_value_fn twice per node.** `_ensure_expanded` called it once for priors, then `_evaluate` called it again for the value — doubling work for every new node. Fix: `_ensure_expanded` now returns the value it gets from `policy_value_fn`; `_evaluate` reuses it and skips the second call.

16. **AlphaBeta as inspector prior is slow.** Every MCTS node expansion calls `ab_agent.choose_move(game)` which runs a full alpha-beta search from that position. Acceptable for analysis use, but Ludacris depths=(20,20,20) will be very slow. Expected behavior — not a bug.

17. **`Move` is not hashable — cannot be used as dict key.** `_make_alphabeta_pv` and `_make_minimax_pv` originally used `{move: prior for move, prior in distribution}`. Fix: match by equality using `next()` over a list, relying on dataclass `__eq__` (field-by-field comparison).

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
**Current state:** mean ≈ -0.01 (good), std ≈ 0.64 (too high — factory memorization).

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

**Medium run — Phase 1 (value calibration, diff-only, with dropout):**
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

---

### Inspector UI

- Agent selector dropdown: Minimax, AlphaBeta Easy/Medium/Hard/Extreme/Ludacris, AlphaZero
- AlphaBeta shows preset depth/threshold parameters as read-only label
- AlphaZero shows sim count dropdown (50/100/200/500/1000/5000)
- Start/Pause toggle, sim count from `root.visits`
- Fully-explored detection — PUCT skips fully-explored nodes
- Per-move immediate score delta via `earned_score_unclamped`
- Cumulative minimax rollup of immediate scores
- Policy prior (`p=X.X%`) per node — network's initial move preference
- Visit count and visit fraction of parent per node
- Net value diff — raw net output before search (meaningful with AlphaZero)
- Children sorted by cumulative score, alternating desc/asc by depth
- Copy Tree button — output includes agent name and parameters in header
- Run server without `--reload` during training

**Minimax value semantics:** `minimax_pv` returns `(current_score - opponent_score) / 50.0` using `earned_score_unclamped`. This is the score differential as a fraction of a typical game spread, evaluated at every node (not just terminals). `net_value_diff` shows this raw evaluation at first visit; `avg` shows the MCTS-averaged value after search. With full exploration, `avg` and `minimax_value` converge.

---

### Agent registry

`agents/registry.py` is the single source of truth. `GET /agents` serves visible agents. Adding a new agent: registry entry + `PlayerType` in `schemas.py`.

UI difficulty levels:
- `alphabeta_easy`: `depths=(1,2,3), thresholds=(20,10)`
- `alphabeta_medium`: `depths=(2,3,7), thresholds=(20,10)`
- `alphabeta_hard`: `depths=(3,5,7), thresholds=(20,10)` — promotion bar
- `alphabeta_extreme`: `depths=(4,6,8), thresholds=(20,10)`
- `alphabeta_ludacris`: `depths=(20,20,20), thresholds=(180,180)`

---

### Open issues

- **Value head overfit to factory configurations** — current blocker. std ≈ 0.64 on empty boards; target < 0.3. Fix: model dropout + smaller hidden_dim.
- **PUCT uses `value_diff` not `value_win`** — evaluate switching once value_diff is calibrated.
- **`_terminal_value` uses clamped `earned_score`** — mismatch with training targets. Low priority.
- **`distinct_pairs /10` not clamped to 1.0** — early round values can exceed 1.0. Low priority.
- **Move cap of 100 in eval** — too low when nets are poorly calibrated. Consider `--eval-move-cap 300`.
- **`net_value_diff` display uses ×20 multiplier** — should be ×50 to match the minimax divisor and training targets. All inspector point values should be on the same scale.

### Deferred

- **Prior round board state snapshots** — one encoded board state per completed round, up to 5.
- **AlphaBeta depth selection via `count_distinct_source_color_pairs()`** — cleaner than legal move count.
- **Parallel self-play via multiprocessing** — heuristic collection now parallel; self-play still sequential.
- **Encoding cache keyed by Zobrist hash** — saves ~19% search time.
- **Shared state tree for two-agent eval** — solves two-tree eval architecturally.
- **Elo ladder** across all agent versions.

### Inspector UI todos

- [ ] **Copy node button** — tiny icon on each row, copies the path from root to that node only (not the full tree). Decide whether to include board state at that position.
- [ ] **Condensed state copy format** — current copy is verbose; make it more compact and human-readable for pasting into prompts or notes.
- [ ] **Load game state from plain text** — paste a copied state string into the UI to jump to that position.
- [ ] **Node IDs in tree display** — show a short identifier on each row for easier reference when discussing specific nodes.
- [ ] **Fix `net_value_diff` display multiplier** — currently ×20, should be ×50 to match minimax divisor and training scale so all point columns are comparable.
- [ ] **Scale indicator in inspector** — small note or tooltip clarifying that all point values are `earned_score_unclamped` differential ÷ 50, displayed as ×50 for readability.

### Next up

- [ ] Add dropout + `--hidden-dim` to `model.py`, rerun Phase 1
- [ ] Confirm `value_diff` std < 0.3 via `sample_policy`
- [ ] Copy `latest.pt` to `gen_0001.pt`, run Phase 2
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
| `MinimaxAgent` | >> Greedy (100%) | Uniform (inherited) | Inspector backend + analysis |
| `AlphaBetaAgent` | >> Minimax (76%) | Softmax over search scores | Pretrain opponent + UI bot + promotion bar + inspector backend |
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
| `scripts/sample_policy.py` | Bulk value calibration across N random game states |
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
| 2026-04-23 | Phase 8e begun. AlphaBeta scored policy distribution. Diverse heuristic matchups. Parallel heuristic collection. Training loop net-reset bug fixed. diff-only mode. Checkpoint management improvements. Diagnostic scripts: inspect_policy, sample_policy. Value head overfit to factory configurations identified (std=0.64, target <0.3). Model dropout needed. Encoder rewritten — all tests passing. |
| 2026-04-25 | Inspector agent selector shipped (Minimax, AlphaBeta presets, AlphaZero). Policy priors and visit fractions added to inspector nodes. visit_fraction added to _serialize_node. Minimax inspector value fixed: earned_score_unclamped ÷ 50, no clamp. Double policy_value_fn call per node eliminated (_ensure_expanded now returns value). Copy Tree includes agent header. Move is not hashable — prior matching uses list scan not dict. Inspector UI todos logged. |