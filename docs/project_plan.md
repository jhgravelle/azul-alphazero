# project_plan.md

# Azul AlphaZero — Project Plan

> Last updated: 2026-05-05
> Status: Phase 8 in progress. 8i complete — encoding v3 (spatial 8→4 channels, flat 8→53 values). All 488 non-engine tests passing. Existing checkpoints are incompatible (conv_local and flat_proj weight shapes changed). Next: 8h inspector serialization redesign or fresh training run with v3.

---

## Vision

Build a fully playable implementation of the board game **Azul** with an **AlphaZero-style AI opponent**, deployable as a web app. Primary goal: deep understanding of the training pipeline. Secondary goal: a shareable web app with AlphaZero as the opponent.

---

## Technology Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.12 | Primary language, best ML ecosystem |
| Front end | FastAPI + HTML/JS | Web-first, shareable by URL |
| Testing | pytest | Industry standard for Python TDD |
| Version control | Git + GitHub | Standard, CI/CD integration |
| CI/CD | GitHub Actions | Free for public repos |
| ML framework | PyTorch | Best for custom AlphaZero-style training loops |
| IDE | VS Code + Claude Code | Installed, good Python + git support |
| OS | Windows 11 25H2 64-bit | |
| GPU | nVidia RTX 5070 | Training runs on GPU |
| CPU | Ryzen 7 7800X3d 8-Core | MCTS inference runs on CPU — CPU is the bottleneck |

**Compute split:** MCTS inference runs on CPU (faster for small batch sizes). Training runs on GPU. GPU sits near-idle during self-play and eval.

---

## AlphaZero Design Decisions & Deviations

This section tracks where our implementation diverges from the original AlphaZero paper and why.

### True AlphaZero (reference)
- 800 MCTS simulations per move, same count for both training and eval games
- Training: Dirichlet noise added to root priors; moves sampled proportionally to visit counts (temperature > 0)
- Evaluation: moves selected greedily by root visit count (temperature → 0, no noise); separate from training games
- Millions of self-play games on thousands of TPUs

### Our Deviations

**Deviation 1 — Simulation count**
True AlphaZero uses 800 simulations per move for both training and eval. Our count is TBD — Azul has a much smaller branching factor than chess/Go, so far fewer simulations may be sufficient. This is a hyperparameter to tune. Current training uses configurable sim counts (50/100/200/500/1000/5000 in the UI).

**Deviation 2 — Training game generation**
True AlphaZero generates training games from the single current best model with exploration enabled. Our current setup TBD — to be clarified and documented here.

**Deviation 3 — Evaluation approach**
True AlphaZero runs a fixed evaluation tournament with exploration off and promotes if win rate exceeds a threshold. We are considering using **model differences as implicit exploration** rather than an explicit exploration-off parameter — two models at different training stages naturally diverge in play. If high game counts and sim counts are not required to determine the better model, the true AlphaZero approach is preferable.

**Deviation 4 — Virtual loss and batching**
Our batched MCTS with virtual loss provides exploration breadth within each batch, partially compensating for any reduction in per-move simulation count. This is an architectural difference from the original sequential MCTS.

**Deviation 5 — Mirror games**
We use mirrored game pairs (same bag seed, both players swap sides) for both training and evaluation. True AlphaZero does not do this — it relies on volume. Our mirror design controls for first-player advantage and the tile draw randomness within a game pair, providing stronger signal per game pair. The bag is only shuffled at game start and at refill; refill does not occur until round 6, which most high-quality games never reach. This guarantees identical factory contents for rounds 1–5 regardless of move choices.

---

## Repository Structure

```
azul-alphazero/
├── agents/
│   ├── alphabeta.py       # Alpha-beta pruning, softmax policy_distribution
│   ├── alphazero.py       # Thin wrapper — delegates to SearchTree
│   ├── base.py            # Agent base class + default policy_distribution (uniform)
│   ├── cautious.py        # Uniform over non-floor moves
│   ├── efficient.py       # Uniform over partial-line moves (fallback to all)
│   ├── greedy.py          # Color-conditional distribution
│   ├── mcts.py
│   ├── minimax.py         # Depth-limited minimax, full tree (no pruning)
│   ├── move_filters.py    # non_floor_moves shared helper
│   ├── random.py          # Inherits uniform distribution
│   └── registry.py        # Single source of truth for all agents
├── api/
│   ├── main.py
│   └── schemas.py
├── engine/
│   ├── constants.py       # Tile enum, WALL_PATTERN, FLOOR_PENALTIES, all constants
│   ├── game.py            # Game controller: make_move, advance, legal_moves, score_round
│   ├── game_recorder.py   # GameRecorder, GameRecord, RoundRecord, MoveRecord
│   ├── player.py          # Player dataclass — owns all board state and scoring
│   └── replay.py          # replay_to_move
├── frontend/
│   ├── game.js
│   ├── index.html
│   ├── render.js
│   └── style.css
├── neural/
│   ├── encoder.py         # Spatial + flat encoding (v3: spatial (4,5,5), flat 53)
│   ├── model.py           # Multi-kernel spatial branch (180k params) — shrink experiment
│   ├── replay.py          # Circular replay buffer, three value targets per example
│   ├── search_tree.py     # SearchTree: MCTS, background worker, subtree reuse
│   ├── trainer.py         # compute_loss, Trainer, target functions, data collection
│   └── zobrist.py         # PENDING DELETION — replaced by move-path node keys
├── scripts/
│   ├── benchmark_agents.py
│   ├── benchmark_mcts.py
│   ├── bench_score_placement.py
│   ├── debug_minimax.py   # One-off diagnostic (virtual loss investigation)
│   ├── inspect_policy.py  # Per-move policy/value/MCTS diagnostic tool
│   ├── migrate_recordings.py
│   ├── parse_log.py
│   ├── sample_policy.py   # Bulk value head calibration checker (N random states)
│   ├── self_play.py
│   ├── tournament.py      # Round-robin parallel tournament; games feed replay buffer
│   └── train.py
├── scratch/
│   └── draft_game         # Draft engine rewrite (Game + Player taking over scoring)
├── tests/
│   ├── engine/
│   │   ├── test_game.py
│   │   └── test_player.py
│   └── [legacy flat test files — move to subfolders when touched]
├── checkpoints/           # gitignored; latest.pt = most recent training state
├── cli/
├── htmlcov/               # gitignored
└── recordings/            # gitignored; human v human games preserved here
```

---

## Development Phases

### Phases 0–7 ✅
Project setup, game engine, frontend, random bot, MCTS, neural network, AlphaZero self-play, reward shaping, undo/hypothetical/manual factory setup. All complete.

---

### Phase 8 — Evaluation and Iteration 🔄 (in progress)

#### 8a — Search and Encoding Rewrite ✅
Spatial+flat encoding, conv+MLP model, Zobrist hashing, game-owned SearchTree with transposition table and subtree reuse, factory canonicalization, round boundaries as leaves.

#### 8b — Batched Multithreaded MCTS ✅
Virtual loss, parallel leaf collection, single batched forward pass per batch.

#### 8c — Heuristic Baseline ✅
AlphaBeta hard wins 76% vs Minimax. Inspector UI complete. AlphaBeta is pretrain opponent and promotion bar.

#### 8d — Encoding Upgrade ✅
Spatial (12,5,6) → (14,5,6). Blocked wall channel, unclamped scores, round progress, distinct pairs.

#### 8e — Training Pipeline + Diagnostics ✅
AlphaBeta scored policy distribution. Diverse heuristic matchups. Parallel heuristic collection. Training loop net-reset bug fixed. diff-only mode. Checkpoint management. Diagnostic scripts. Inspector agent selector, policy priors, visit fractions. Minimax inspector value fixed. Double policy_value_fn call eliminated.

#### 8f — Engine Rewrite ✅

**Goal:** `Player` owns all scoring. `Game` owns round/game transitions and calls out to players. `scoring.py`, `board.py`, `game_state.py` deleted.

Key changes:
- `player.earned` (cached property: `score + pending + penalty + bonus`) replaces all `earned_score_unclamped()` callsites
- `player.handle_round_end()` replaces `Game._score_floor()` and wall placement logic
- `player.handle_game_end()` applies end-of-game bonuses
- `Game.score_round()` calls `player.handle_round_end()` for each player
- `Game.score_game()` calls `player.handle_game_end()` for each player
- Pure scoring functions in `scoring.py` become private methods on `Player`
- `score_placement` moves to `Player` (operates on the wall)
- `_handle_round_end()` in `api/main.py` returns `(round_ended: bool, game_ended: bool)`
- `count_distinct_source_tile_pairs()` removed from `Game` (never hooked up)
- `Game.determine_winners() -> list[Player]` — returns list of one or two players. Tiebreak: (1) highest score after bonuses, (2) most completed horizontal wall rows, (3) shared victory. No column tiebreaker — not in official rules. Replaces wherever winner determination currently lives in the API or frontend.
- Deleted: `engine/board.py`, `engine/game_state.py`, `engine/scoring.py`
- Deleted associated tests: `test_board.py`, `test_game_state.py`, `test_scoring.py`
- Moved `tests/test_game.py` → `tests/engine/test_game.py`

**Status:** Complete. 625 tests passing.

#### 8g — Virtual Loss / MCTS Bug Fix ✅

**Bug:** During a search, all root children get one visit each (correct). All remaining simulations then pile onto one branch despite virtual loss being applied.

**Root cause:** `_select_vl` passed `node.visits` as `parent_visits` to `puct_score`. During batch collection, backprop hasn't run yet, so `node.visits` is 0 for the root the entire batch. This makes the exploration term `U = C * prior * sqrt(0) / (1 + adj_visits) = 0` for every child. With U=0, PUCT collapses to Q-only selection. After all children have been visited once, the highest-prior child wins every subsequent selection in the batch.

**Fix:** In `_select_vl`, increment `virtual_loss` before computing `parent_visits`, then pass `node.visits + node.virtual_loss`:

```python
node.virtual_loss += 1
parent_visits = node.visits + node.virtual_loss
node = max(eligible, key=lambda c: c.puct_score(parent_visits))
```

**Confirmed by:** `inspect_policy.py` MCTS probe with `batch_size=250`. Before fix: top move got 261/500 visits (52%). After fix: top move got 28/500 visits (5.6%), spread proportional to priors.

#### 8h — Inspector Serialization Redesign

**Root cause of current bug:** `serialize(max_depth=4)` silently cuts the tree at depth 4. In a late-round position with 4 moves already played, the final move's children fall at depth 5 and disappear. The node appears as a childless leaf and its `cumulative_immediate` is computed as if terminal, corrupting the rollup up the chain.

**Redesign:**

**1. Tree owns immediate and cumulative values.**
`AZNode` gets two new fields: `immediate: float | None` and `cumulative_immediate: float | None`. Set during backpropagation or a post-pass after each batch. `_serialize_node` reads them — no inline rollup logic. Cumulative is a minimax rollup: max at root player's turns, min at opponent's, from root player's perspective.

**2. Node keys use move-path, not Zobrist hash.**
Each node's key is the sequence of move indices from the start of the round. Max 20 moves per round. One byte per move index. Key is ~10 bytes maximum. Zobrist hash and `neural/zobrist.py` are removed.

**3. Background MCTS worker.**
MCTS runs continuously in a background thread. The poll endpoint snapshots current tree state at whatever depth exists. Poll interval controls UI refresh rate, not tree growth rate.

**4. Serialization depth.**
`tree.serialize()` removes the `max_depth` cap. All nodes included — round boundary naturally limits depth to ≤20 moves.

**5. Polling payload.**
Response includes root node + nodes up to 2 levels deep. Nodes at the display boundary that have children carry `"has_more": true`.

**6. Subtree fetch endpoint.**
```
GET /inspect/subtree?key=<move_path_hex>&depth=3
```

**7. Chevron click behavior.**
- Children already in payload: expand client-side instantly.
- Node has `"has_more": true`: trigger subtree fetch.

**Changes required:**

| Location | Change |
|---|---|
| `AZNode` | Add `immediate`, `cumulative_immediate` fields |
| `SearchTree._backpropagate` | Compute and store immediate + cumulative on each node |
| `SearchTree.serialize()` | Remove depth cap; read stored values |
| `SearchTree._serialize_node` | Simplify — read stored fields, add `has_more` flag |
| `neural/zobrist.py` | Delete |
| API polling endpoint | Pass `max_depth=None`; return 2-level payload with `has_more` |
| New API endpoint | `GET /inspect/subtree?key=...&depth=3` |
| Frontend `renderInspectorPanel` | Chevron triggers subtree fetch on `has_more=true` |

#### 8q — Factory Overfitting Mitigation via Mirror Games ✅

**Problem:** Value head overfits to factory state. Some games have correlated outcomes (e.g., games with larger factories cluster in wins), allowing the value head to memorize factory fingerprints rather than learn pure board evaluation.

**Root cause:** Non-mirrored games introduce factory-outcome correlation. If one agent is stronger, games with certain factory distributions may systematically favor that player, creating spurious patterns.

**Solution:** Mirror game pairs — identical bag seed, both players swap sides.
- Game 1: Agent A vs Agent B, factories F
- Game 2: Agent B vs Agent A, same factories F

Both games use `Game(seed=N)` to guarantee identical factory sequences. Since the bag has exactly 100 tiles and no refill occurs until round 6, the seed fully determines all factories for rounds 1–5.

**If agent A is stronger:** A wins from both sides, so factory state didn't determine outcome — board state did. This forces the value head to learn board evaluation.

**Results (20 iterations, 100 mirror pairs per iter, 2000 games total):**
- `value_win` std: 0.0409 (tight, no factory noise)
- `value_diff` std: 0.0380 (tight, no factory noise)  
- `value_abs` std: 0.0295 (tight, no factory noise)

All three heads converge cleanly without factory fingerprinting.

**Implementation:**
- Use `collect_mirror_heuristic_games()` in trainer.py — already implemented
- Add `--mirror-games-per-iter N` to training loop
- Recommendation: 20–50 mirror pairs per iteration once model reaches competence
- **Apply to all agent vs agent games:** self-play, heuristic collections, eval tournaments

**Changes required:**

| Location | Change |
|---|---|
| `scripts/train.py` | Already supports `--mirror-games-per-iter`; document best practices |
| `neural/trainer.py` | Already implemented; ensure used in all data collection |
| Documentation | Add mirror games as standard practice for all matchups |

#### 8i — Encoding v3 ✅

**Motivation:** Current encoding mixes non-spatial data into the spatial tensor where conv2d cannot use it effectively.

**Spatial tensor: `(4, 5, 5)`** — down from `(14, 5, 5)`

| Channel | Content |
|---|---|
| 0 | My wall filled |
| 1 | My pattern line fill ratio |
| 2 | Opponent wall filled |
| 3 | Opponent pattern line fill ratio |

**Flat vector: 53 values**

| Offset | Count | Name | Encoding |
|---|---|---|---|
| 0 | 1 | My official score | ÷ 100 |
| 1 | 1 | Opponent official score | ÷ 100 |
| 2 | 1 | My earned | ÷ 100 |
| 3 | 1 | Opponent earned | ÷ 100 |
| 4 | 1 | My floor penalty | ÷ 14 |
| 5 | 1 | Opponent floor penalty | ÷ 14 |
| 6 | 1 | I hold first-player token | 0 or 1 |
| 7 | 1 | Opponent holds FP token | 0 or 1 |
| 8 | 5 | My row completion | weighted ÷ ((row+1)×5) per row |
| 13 | 5 | Opponent row completion | weighted ÷ ((row+1)×5) per row |
| 18 | 5 | My col completion | weighted ÷ 15 per col |
| 23 | 5 | Opponent col completion | weighted ÷ 15 per col |
| 28 | 5 | My color completion | weighted ÷ 15 per color |
| 33 | 5 | Opponent color completion | weighted ÷ 15 per color |
| 38 | 5 | Tiles available by color | count ÷ 20 |
| 43 | 5 | Sources with color | count ÷ 5 |
| 48 | 5 | Bag count by color | count ÷ 20 |

Net input: 4×5×5 spatial (100 values) + 53 flat = 153 total, down from 208.

#### 8j — Training Pipeline Phase 1 + Phase 2

**Phase 1 — Value calibration (diff-only)**

Goal: `value_diff` std < 0.3 on turn-1 empty boards.

```
python -m scripts.train \
  --iterations 30 --games-per-iter 0 --simulations 200 \
  --train-steps 1000 --value-only-iterations 0 \
  --skip-eval-iterations 30 --eval-games 0 --eval-simulations 200 \
  --win-threshold 0.55 --alphabeta-games-per-iter 100 \
  --buffer-size 500000 --heuristic-workers 8 --diff-only
```

**Phase 2 — Full training**

```
python -m scripts.train \
  --iterations 30 --games-per-iter 10 --simulations 200 \
  --train-steps 1000 --value-only-iterations 0 \
  --skip-eval-iterations 5 --eval-games 20 --eval-simulations 200 \
  --win-threshold 0.55 --alphabeta-games-per-iter 40 \
  --candidate-games-per-iter 20 --buffer-size 200000 \
  --heuristic-workers 8 --initial-generation 1
```

**Heuristic matchup weights:**

| Matchup | Weight | Purpose |
|---|---|---|
| Random vs AlphaBeta easy | 10% | Extreme loss signal |
| Efficient vs AlphaBeta easy | 20% | Weak vs strong |
| Cautious vs AlphaBeta easy | 30% | Moderate loss signal |
| Greedy vs AlphaBeta easy | 40% | Near-peer, clean policy targets |

Note: easy = `depths=(1,1,3)` (weakened from `(2,3,7)` for experiment). Easy-vs-easy dropped (was 45%).

**Value heads:**
- `value_diff` — primary PUCT signal
- `value_win` — auxiliary, weight 0.3
- `value_abs` — weight 0.1; candidate for removal

**Promotion bar:** Beat `alphabeta_hard` at ≥55% win rate with ≥1500 eval simulations.

**Graduated eval targets:**
1. Beat Greedy ≥70%
2. Beat Cautious ≥60%
3. Beat AlphaBeta easy ≥55%
4. Beat AlphaBeta hard ≥55% — deployable

#### 8k — Elo Ladder

- Elo tracked live during training, logged per iteration
- Tournament games double as training data
- Goal: UI page showing Elo ladder across all `gen_xxxx.pt` checkpoints
- Multiple model sizes trained on same data; Elo is the selection mechanism

#### 8l — Evaluation & Variance Analysis

**Goal:** Once a strong model exists, quantify how often the stronger player wins to understand Azul's luck/variance profile.

**Approach:**
- Head-to-head tournaments between strong model and weaker baselines
- All eval games use mirrored pairs (same bag seed, players swap sides) — same discipline as training
- Aggregation options per pair: count games independently, win-both/split/lose-both, or net score differential across the pair
- Score differential may be more informative than binary win/loss when models are close in strength
- Run enough game pairs for statistical confidence (target 500–1000+ games)

**Open questions:**
- How many simulations are needed for eval results to be meaningful? (Current guidance: ≥1500)
- Does the stronger model need to be AlphaZero-level, or is AlphaBeta hard vs AlphaBeta medium sufficient to measure variance?

#### 8m — Parallel Eval Games

- Confirm or implement parallel execution of eval games
- Individual games always run to completion; early exit applies only at the tournament level (stop if candidate has clinched promotion or been eliminated)
- Parallelism makes full-game eval affordable even without early exit

#### 8n — Multi-Flavor Hyperparameter Search (Early Stage)

**Goal:** Run multiple model flavors with differing hyperparameters in parallel during early training to identify the best model structure before committing to a full training run.

**Design:**
- Each flavor has its own current best model
- A candidate promotes within its flavor by beating its own prior best
- Cross-flavor eval games are used for training data richness and gameplay diversity only — they do not affect promotion decisions
- Each eval phase tournament includes best and candidate models from all active flavors
- Once a winning structure is identified, commit to it for the full training run

**Hyperparameters to vary:** model architecture (depth, width), PUCT constant, simulation count, encoding choices. Treat as a hyperparameter; Elo selects the winner.

#### 8o — Training Dashboard & Remote Monitoring

**Goal:** Web dashboard for monitoring training progress, accessible remotely (e.g. from phone while away from the computer).

**Metrics to display:**
- Loss curves (policy head, value heads, total)
- Elo progression across checkpoints
- Win rates in eval tournaments per flavor/checkpoint
- Current tournament standings and games remaining
- Self-play health: games per hour, GPU utilization, replay buffer size

**Implementation options:**
- Lightweight local web page served by the existing FastAPI server
- Weights & Biases (wandb) — handles remote access automatically, good mobile viewing

---

### Phase 9 — Polish and Release (wishlist)

- Animated tile placement
- Sound effects
- Cloud deployment
- README with screenshots
- Capacitor iOS/Android packaging (low priority)

---

## Backlog

- [ ] Write mirror game invariant test — confirm identical factory contents rounds 1–5 with same bag seed using random agents; assert round 6 is first potential divergence point. Acts as regression guard against future bag/shuffle changes.
- [ ] **Model shrinking phase 2** — Current shrunk model is 706k params (2.1× reduction). Investigate further shrinking to ~350k params by: reducing ResBlocks from 3→1 (saves 262k), reducing hidden_dim from 256→192, reducing spatial channels from 18→14. Test value head calibration at each reduction to find the floor before overfitting returns.
- [ ] Confirm or implement parallel eval games (see 8m)
- [ ] Add `--no-early-exit` argument to eval tournament, or document current behavior clearly
- [ ] Start design decisions doc — AlphaZero deviations (covered above in "AlphaZero Design Decisions & Deviations" section)
- [ ] Fix `net_value_diff` display multiplier — currently ×20, should be ×50
- [ ] Copy node button — tiny icon per row, copies path from root to that node only
- [ ] Condensed state copy format — compact and human-readable for pasting into prompts
- [ ] Load game state from plain text — paste a copied state string to jump to that position
- [ ] Scale indicator — tooltip clarifying all point values are `earned` differential ÷ 50, displayed ×50

---

## Engine Design Reference

### make_move / advance separation

```
make_move(move)              — take tiles from source, place on pattern line/floor;
                               update player scoring caches
advance(skip_setup=False)    — next_player(), score_round() if round over,
                               score_game() if game over, setup_round() unless
                               skip_setup=True. Returns True if round boundary crossed.
is_round_over() → bool       — True when no color tiles remain in any source
score_round()                — calls player.handle_round_end() for each player,
                               sets next first player
is_game_over() → bool        — True if any player has a completed wall row
score_game()                 — calls player.handle_game_end() for each player
setup_round()                — fill factories, add FIRST_PLAYER to center
```

### Player scoring model

```
player.score      — confirmed score from prior rounds (always >= 0)
player.pending    — placement points from full pattern lines this round
player.penalty    — floor penalty this round (negative or zero)
player.bonus      — end-of-game bonus points (updated when wall changes)
player.earned     — score + pending + penalty + bonus (cached property, never stored)
```

`player.earned` is unclamped mid-round. Clamping happens only in `player.handle_round_end()`.

Scoring divisors: absolute score ÷ 100, score differential ÷ 50.

### API round transitions

`_handle_round_end()` in `api/main.py` is the single entry point after every `make_move`. Returns `(round_ended: bool, game_ended: bool)`. Never call `advance()` directly in API code.

---

## Agent Reference

| Agent | Strength | `policy_distribution` | Purpose |
|---|---|---|---|
| `RandomAgent` | Baseline | Uniform over legal (inherited) | Benchmark floor |
| `EfficientAgent` | ~22% overall | Uniform over partial-line | Weak — too passive |
| `CautiousAgent` | ~47% vs Greedy | Uniform over non-floor | Avoids penalties |
| `GreedyAgent` | ~49% overall | Color-conditional | Training opponent (weak side) |
| `MCTSAgent` | Untested vs new agents | N/A | Lookahead without neural net |
| `MinimaxAgent` | >> Greedy (100%) | Uniform (inherited) | Inspector backend + analysis |
| `AlphaBetaAgent` | >> Minimax (76%) | Softmax over search scores | Pretrain opponent + UI bot + promotion bar |
| `AlphaZeroAgent` | Goal: >> AlphaBeta hard | Via SearchTree MCTS visits | Final goal |

### AlphaBeta depth configuration

```python
AlphaBetaAgent(depths=(d1, d2, d3), thresholds=(t1, t2))
# legal_count > t1  -> depth d1
# t2 < legal_count <= t1 -> depth d2
# legal_count <= t2 -> depth d3
```

UI difficulty levels:
- `alphabeta_easy`: `depths=(1,2,3), thresholds=(20,10)`
- `alphabeta_medium`: `depths=(2,3,7), thresholds=(20,10)`
- `alphabeta_hard`: `depths=(3,5,7), thresholds=(20,10)` — promotion bar
- `alphabeta_extreme`: `depths=(4,6,8), thresholds=(20,10)`
- `alphabeta_ludacris`: `depths=(20,20,20), thresholds=(180,180)`

---

## Inspector UI Reference

### Current capabilities
- Agent selector: Minimax, AlphaBeta Easy/Medium/Hard/Extreme/Ludacris, AlphaZero
- AlphaBeta shows preset depth/threshold parameters as read-only label
- AlphaZero shows sim count dropdown (50/100/200/500/1000/5000)
- Start/Pause toggle; sim count from `root.visits`
- Per-move immediate score delta (`player.earned` differential)
- Cumulative minimax rollup of immediate scores
- Policy prior (`p=X.X%`) per node
- Visit count and visit fraction of parent per node
- Net value diff — raw net output before search
- Children sorted by cumulative score, alternating desc/asc by depth
- Copy Tree — includes agent name and parameters in header
- Run server without `--reload` during training

### Minimax value semantics
`(current_player.earned - opponent.earned) / 50.0` at every node.

### Inspector UI active work items
- [ ] Fix `net_value_diff` display multiplier — currently ×20, should be ×50
- [ ] Copy node button — tiny icon per row, copies path from root to that node only
- [ ] Condensed state copy format — compact and human-readable
- [ ] Load game state from plain text
- [ ] Node IDs — falls out naturally from move-path key scheme (see 8h)
- [ ] Scale indicator — tooltip clarifying all point values are `earned` differential ÷ 50

---

## Multi-head Value Network

`AzulNet.forward(spatial, flat)` returns `(logits, value_win, value_diff, value_abs)`.

```python
logits, value_win, _value_diff, _value_abs = net(spatial, flat)
```

---

## Checkpoint Management

- `checkpoints/latest.pt` — current training weights
- `checkpoints/latest_params.json` — args used to produce `latest.pt`
- `checkpoints/gen_xxxx.pt` — promoted checkpoints only
- `--load` defaults to `checkpoints/latest.pt`; silently skips if missing
- `--initial-generation N` — start generation counter at N

---

## Hard-Won Lessons (Do Not Repeat)

1. **`value_only_iterations > 0` is a divergence trap.** Always `--value-only-iterations 0`.
2. **One-hot policy targets poison the policy head.** Fix: `policy_distribution()` returns soft distributions.
3. **Random agent pretrain is nearly useless.** Floor-heavy games, near-zero score signal.
4. **Clearing buffer after pretrain kills value signal.** Never use `--clear-buffer-after-pretrain`.
5. **Eval at low sim counts is nearly useless.** Need ≥1500 sims for meaningful eval.
6. **Net weights reset every iteration when eval was skipped.** Fix: only reset when eval actually ran and net lost.
7. **Policy loss dominates the trunk (~50x value loss).** Use `--diff-only` Phase 1.
8. **Value head overfit to factory configurations.** Fix: encoding v3 removes raw factory distribution.
9. **MCTS snowballs on high-value outliers.** Fix: value head calibration (Phase 1).
10. **Eval against random `gen_0000.pt` produces move-cap hits.** Fix: copy `latest.pt` to `gen_0001.pt` after Phase 1.
11. **`uvicorn --reload` restarts server on checkpoint writes.** Run without `--reload` during training.
12. **`_ensure_expanded` called policy_value_fn twice per node.** Fix: `_ensure_expanded` returns the value; `_evaluate` reuses it.
13. **AlphaBeta as inspector prior is slow.** Expected behavior — not a bug.
14. **`Move` is not hashable.** Cannot be used as dict key. Match by equality using `next()` over a list.
15. **Virtual loss needs `visits + virtual_loss` as parent count in PUCT.** Passing only `visits` makes U=0 during batch collection, collapsing exploration to prior-greedy selection within each batch.
16. **`while not game.is_game_over()` as a loop condition is always wrong** — `is_game_over()` now requires both `has_triggered_game_end()` AND `is_round_over()`. Use `while not game.is_game_over()` only after confirming this is what you mean. In practice, game loops should check `is_game_over()` after `advance()`, not before `make_move()`.
17. **`has_triggered_game_end()` does not mean the game is over** — it means the game will end at the end of the current round. Moves may still remain this round. `is_game_over()` is only True when the round has also fully ended and all scoring is complete.

---

## Deferred / Future Ideas

- **Parallel self-play via multiprocessing** — heuristic collection is parallel; self-play still sequential.
- **Encoding cache / state reuse** — cache `(encoded_state, legal_moves)` per node within a search.
- **Shared game-owned state tree** — game owns a cache of encoded states; agents request batches directly.
- **Node-budget AlphaBeta** — cap total nodes explored rather than depth.
- **Prior round board state snapshots** — one encoded state per completed round, up to 5.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/train.py` | AlphaZero self-play training loop |
| `scripts/inspect_policy.py` | Per-move policy/value/MCTS diagnostic |
| `scripts/sample_policy.py` | Bulk value calibration across N random game states |
| `scripts/tournament.py` | Round-robin parallel tournament |
| `scripts/benchmark_agents.py` | First-move vs overall timing by depth config |
| `scripts/benchmark_mcts.py` | MCTS throughput benchmarking |
| `scripts/self_play.py` | Generate self-play games |
| `scripts/parse_log.py` | Parse training logs |
| `scripts/migrate_recordings.py` | Migrate old recording format |
| `scripts/debug_minimax.py` | One-off diagnostic (virtual loss investigation) |

---

## Change Log

| Date | Change |
|---|---|
| 2026-05-05 | **8f cleanup complete:** Deleted `engine/board.py`, `engine/scoring.py`, `engine/game_state.py` and their test files. Moved `tests/test_game.py` → `tests/engine/test_game.py`. 625 tests passing. |
| 2026-05-05 | **AzulNet 12x shrink experiment:** Replaced 2-conv spatial branch (1.5M params) with multi-kernel design — local 5×5 (10ch) + row 1×5 (4ch) + col 5×1 (4ch) = 18ch, ~180k params. Bottleneck: Linear(450→256) vs prior Linear(3200→256). Trainer matchups rebalanced: easy-vs-easy dropped (0.45→0.00), weight shifted to greedy/cautious/efficient. AlphaBeta weakened to `depths=(1,1,3)` for experiment. Fixed `sample_policy.py`: `game.state.current_player` → `game.current_player`. |
| 2026-05-04 | **MCTSAgent 3.5× speedup:** `clone()` replaces `deepcopy`, round-boundary rollouts, 200 sims default. |
| 2026-05-05 | **Mirror games discovery:** Value head overfitting to factory state resolved by using mirrored game pairs (identical bag seed, sides swapped) in all agent vs agent collections. Testing shows `value_win`, `value_diff`, `value_abs` stds all < 0.041 on turn 1 (clean calibration). Added 8q phase documenting solution. Recommended for all future data collection. |
| 2026-05-05 | **Model shrinking experiment:** Tested 706k-param shrunk model (2.1× smaller than original 1.5M). Bottleneck reduced from 819k to 115k params. Without mirror games, shrunk model overfit worse (std +17×). With mirror games, shrunk model achieves same tight calibration as larger model. Indicates overfitting was encoder + data correlation, not parameter count. Further shrinking to 350k params proposed for phase 8q. |
| 2026-05-05 | **Encoding v3 (8i):** Spatial (8,5,5) → (4,5,5) — dropped bonus-proximity, bag, source-dist channels; kept wall + pattern for both players. Flat 8 → 53 — added row/col/color completion (post-placement wall), tiles-available, sources-with-color, bag-count per color. Earned divisor changed 50→100. All 488 non-engine tests passing. Existing checkpoints incompatible. |
| 2026-04-25 | Inspector agent selector, policy priors, visit fractions. Minimax value fixed. Double policy_value_fn call eliminated. Copy Tree includes agent header. Inspector UI todos logged. |
| 2026-05-02 | tests/engine/test_player.py added — 100% public interface coverage. Test subfolder structure introduced. |
| 2026-05-02 | Project plan and instructions overhauled. Encoding v3 designed. Engine rewrite drafted. Virtual loss bug identified as top priority. Inspector serialization redesign fully specified. |
| 2026-05-03 | Virtual loss MCTS concentration bug fixed. Root cause: `node.visits` passed as parent_visits, collapsing exploration. Fix: pass `node.visits + node.virtual_loss`. |
| 2026-05-03 | `inspect_policy.py` probe fixed: removed stale v1 encoder imports, added `--batch-size` arg. |
| 2026-05-03 | Added AlphaZero deviations section, variance/evaluation analysis phase (8l), parallel eval phase (8m), multi-flavor hyperparameter search phase (8n), training dashboard phase (8o). Added mirror game invariant test to backlog. File name comment added to code style preferences. File update instruction added to working preferences. |
| 2026-05-03 | Engine rewrite (8f) complete — Player owns all scoring, Game owns transitions. Move moved into game.py. Board/GameState/scoring.py no longer imported. All 695 non-slow tests passing. |
| 2026-05-03 | Fixed stale game.state.* references in scripts/train.py and trainer.py. Fixed value function type annotations (list[int] → list[float]). |
| 2026-05-03 | Fixed is_game_over() — now requires is_round_over() AND has_triggered_game_end(). Fixed premature game termination in mcts.py, self_play.py, trainer.py, train.py. All slow tests passing. |

> For full history, see `git log`.