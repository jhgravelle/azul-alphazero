# Azul AlphaZero — Project Plan

> Last updated: 2026-05-02
> Status: Phase 8 in progress. Active priority: engine rewrite (8f). Virtual loss bug resolved. Encoding v3 designed, pending engine rewrite. Inspector serialization redesign pending.

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
│   ├── replay.py          # replay_to_move
│   ├── board.py           # PENDING DELETION — superseded by player.py
│   ├── game_state.py      # PENDING DELETION — superseded by game.py/player.py
│   └── scoring.py         # PENDING DELETION — folds into player.py
├── frontend/
│   ├── game.js
│   ├── index.html
│   ├── render.js
│   └── style.css
├── neural/
│   ├── encoder.py         # Spatial + flat encoding (v2 current, v3 planned)
│   ├── model.py           # Conv+MLP trunk with 3 value heads + dropout
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

#### 8f — Engine Rewrite 🔄 (next)

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
- Delete: `engine/board.py`, `engine/game_state.py`, `engine/scoring.py`
- Delete associated tests: `test_board.py`, `test_game_state.py`, `test_scoring.py`

Draft in `scratch/draft_game`.

#### 8g — Virtual Loss / MCTS Bug Fix ✅

**Bug:** During a search, all root children get one visit each (correct). All remaining simulations then pile onto one branch despite virtual loss being applied.

**Root cause:** `_select_vl` passed `node.visits` as `parent_visits` to `puct_score`. During batch collection, backprop hasn't run yet, so `node.visits` is 0 for the root the entire batch. This makes the exploration term `U = C * prior * sqrt(0) / (1 + adj_visits) = 0` for every child. With U=0, PUCT collapses to Q-only selection. After all children have been visited once, the highest-prior child wins every subsequent selection in the batch.

**Fix:** In `_select_vl`, increment `virtual_loss` before computing `parent_visits`, then pass `node.visits + node.virtual_loss`:

```python
node.virtual_loss += 1
parent_visits = node.visits + node.virtual_loss
node = max(eligible, key=lambda c: c.puct_score(parent_visits))
```

This makes the exploration bonus grow as the batch accumulates selections through a node, enabling genuine diversity across the batch.

**Confirmed by:** `inspect_policy.py` MCTS probe with `batch_size=250`. Before fix: top move got 261/500 visits (52%). After fix: top move got 28/500 visits (5.6%), spread proportional to priors.

**Note:** The concentration was originally misdiagnosed as a virtual loss magnitude or batch-size problem. A secondary misdiagnosis came from the probe itself using `batch_size=args.mcts_sims`, which also caused concentration for unrelated reasons. Both are now fixed.

#### 8h — Inspector Serialization Redesign

**Root cause of current bug:** `serialize(max_depth=4)` silently cuts the tree at depth 4. In a late-round position with 4 moves already played, the final move's children fall at depth 5 and disappear. The node appears as a childless leaf and its `cumulative_immediate` is computed as if terminal, corrupting the rollup up the chain.

**Redesign:**

**1. Tree owns immediate and cumulative values.**
`AZNode` gets two new fields: `immediate: float | None` and `cumulative_immediate: float | None`. Set during backpropagation or a post-pass after each batch. `_serialize_node` reads them — no inline rollup logic. Cumulative is a minimax rollup: max at root player's turns, min at opponent's, from root player's perspective.

**2. Node keys use move-path, not Zobrist hash.**
Each node's key is the sequence of move indices from the start of the round. Max 20 moves per round (theoretical maximum: 5 factories × 4 unique tiles, each taken one at a time). One byte per move index. Key is ~10 bytes maximum. Zobrist hash and `neural/zobrist.py` are removed.

**3. Background MCTS worker.**
MCTS runs continuously in a background thread. The poll endpoint snapshots current tree state at whatever depth exists. Poll interval controls UI refresh rate, not tree growth rate. Decouples search speed from request/response cycle.

**4. Serialization depth.**
`tree.serialize()` removes the `max_depth` cap. All nodes included — round boundary naturally limits depth to ≤20 moves. Frontend filters to reachable nodes only. Stale keys (from before root promotion) are discarded — frontend tree state resets on root promotion.

**5. Polling payload.**
Response includes root node + nodes up to 2 levels deep. Nodes at the display boundary that have children in the tree carry `"has_more": true`. Full tree data lives on the backend.

**6. Subtree fetch endpoint.**
```
GET /inspect/subtree?key=<move_path_hex>&depth=3
```
Returns subtree rooted at that node, 3 levels deep. Frontend merges into local tree data and re-renders.

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

#### 8i — Encoding v3

**Motivation:** Current encoding mixes non-spatial data into the spatial tensor where conv2d cannot use it effectively. Moving to flat gives the MLP direct access and reduces spatial noise.

**Spatial tensor: `(4, 5, 5)`** — down from `(14, 5, 5)`

| Channel | Content |
|---|---|
| 0 | My wall filled |
| 1 | My pattern line fill ratio |
| 2 | Opponent wall filled |
| 3 | Opponent pattern line fill ratio |

Two conv layers give a 5×5 receptive field — sufficient to see full rows and columns.

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

Completion metrics use `r+1` weighting — a filled cell in row r contributes `r+1` to its row, column, and color totals, reflecting actual tile cost. Center counts as a source for "sources with color."

**Design notes:**
- Tile-color completion is diagonal-and-wrapped across the wall — the conv cannot easily learn this. Keeping all three completion axes (row, col, color) in flat lets the MLP learn strategic weighting directly.
- `sum(sources_with_color[i])` gives the distinct source-color pair count — sufficient signal for AlphaBeta depth selection without a separate method on `Game`.
- Net input: 4×5×5 spatial (100 values) + 53 flat = 153 total, down from 208.
- Factory fingerprinting risk is eliminated — model never sees raw factory tile distribution, only aggregate availability vectors.

**Model size:** Treat as a hyperparameter. Train multiple sizes on the same data; Elo picks the winner. Shallow conv (2 layers, small flatten) + 128-dim MLP with 2 hidden layers is the starting point. The game is simple enough that a small model should suffice.

#### 8j — Training Pipeline Phase 1 + Phase 2

**Phase 1 — Value calibration (diff-only)**

Goal: `value_diff` std < 0.3 on turn-1 empty boards. Eliminates factory fingerprinting before full training begins.

```
python -m scripts.train \
  --iterations 30 --games-per-iter 0 --simulations 200 \
  --train-steps 1000 --value-only-iterations 0 \
  --skip-eval-iterations 30 --eval-games 0 --eval-simulations 200 \
  --win-threshold 0.55 --alphabeta-games-per-iter 100 \
  --buffer-size 500000 --heuristic-workers 8 --diff-only
```

Check calibration:
```
python -m scripts.sample_policy --checkpoint checkpoints/latest.pt --samples 100 --summary-only
```
Target: `value_diff` mean < |0.1|, std < 0.3. If passing, promote and start Phase 2.

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
| Random vs AlphaBeta medium | 10% | Extreme loss signal |
| Efficient vs AlphaBeta medium | 10% | Weak vs strong |
| Cautious vs AlphaBeta medium | 15% | Moderate loss signal |
| Greedy vs AlphaBeta medium | 20% | Near-peer, clean policy targets |
| AlphaBeta easy vs medium | 25% | Peer matchup |
| AlphaBeta medium vs medium | 20% | Symmetric high quality |

**Value heads:**
- `value_diff` (normalized score differential ÷ 50) — primary PUCT signal. Dense continuous signal, almost always aligns with win/loss.
- `value_win` (+1/0/-1) — auxiliary head, weight 0.3. Inspector shows both during training; long-term we pick one.
- `value_abs` (normalized absolute score ÷ 100) — weight 0.1; candidate for removal.

Loss weights (full training): `policy + 0.3·value_win + 1.0·value_diff + 0.1·value_abs`
Loss weights (`--diff-only`): `value_diff` only.

**100-move cap:** Games hitting the cap are a diagnostic — both players get loss signal. Common cap hits mean the net hasn't learned. They are not a bug. 100 moves is correct; human games max at ~65.

**Promotion bar:** Beat `alphabeta_hard` at ≥55% win rate with ≥1500 eval simulations.

**Graduated eval targets:**
1. Beat Greedy ≥70%
2. Beat Cautious ≥60%
3. Beat AlphaBeta easy ≥55%
4. Beat AlphaBeta hard ≥55% — deployable

#### 8k — Elo Ladder

**Priority: medium-high.**

- Elo tracked live during training, logged per iteration (minimum viable)
- Tournament games double as training data — weak play mixed in helps strong agents remember why bad play is bad
- Goal: UI page showing Elo ladder across all `gen_xxxx.pt` checkpoints from training logs
- Multiple model sizes trained on same data; Elo is the selection mechanism

---

### Phase 9 — Polish and Release (wishlist)

- Animated tile placement
- Sound effects
- Cloud deployment
- README with screenshots
- Capacitor iOS/Android packaging (low priority)

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

`Player` owns all scoring state. `Game` tells players when to score.

```
player.score      — confirmed score from prior rounds (always >= 0)
player.pending    — placement points from full pattern lines this round
player.penalty    — floor penalty this round (negative or zero)
player.bonus      — end-of-game bonus points (updated when wall changes)
player.earned     — score + pending + penalty + bonus (cached property, never stored)
```

`player.earned` is unclamped mid-round — a negative value signals that floor moves are bad. Clamping happens only in `player.handle_round_end()` when `player.score` is set for the next round.

Scoring divisors: absolute score ÷ 100, score differential between players ÷ 50.

### API round transitions

`_handle_round_end()` in `api/main.py` is the single entry point after every `make_move`. It calls `game.advance()` and returns `(round_ended: bool, game_ended: bool)`. Never call `advance()` directly in API code.

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

Depth selected by legal move count at each node:

```python
AlphaBetaAgent(depths=(d1, d2, d3), thresholds=(t1, t2))
# legal_count > t1  -> depth d1 (early round, high branching)
# t2 < legal_count <= t1 -> depth d2
# legal_count <= t2 -> depth d3
```

UI difficulty levels:
- `alphabeta_easy`: `depths=(1,2,3), thresholds=(20,10)`
- `alphabeta_medium`: `depths=(2,3,7), thresholds=(20,10)`
- `alphabeta_hard`: `depths=(3,5,7), thresholds=(20,10)` — promotion bar
- `alphabeta_extreme`: `depths=(4,6,8), thresholds=(20,10)`
- `alphabeta_ludacris`: `depths=(20,20,20), thresholds=(180,180)`

### AlphaBeta policy distribution

`policy_distribution()` returns softmax over root move scores (temperature=1.0). `_score_all_root_moves` evaluates every root move with full alpha/beta window (no root-level pruning). `choose_move` must be called before `policy_distribution` — populates `_root_move_scores` cache. Falls back to uniform if cache empty.

### Agent registry

`agents/registry.py` is the single source of truth. `GET /agents` serves visible agents to the frontend. Adding a new agent requires a registry entry and a `PlayerType` update in `api/schemas.py`. These are currently two separate locations due to circular import constraints — investigate consolidation when the opportunity arises.

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
`(current_player.earned - opponent.earned) / 50.0` at every node. `net_value_diff` shows raw evaluation at first visit; `avg` shows MCTS-averaged value after search. With full exploration, `avg` and minimax value converge.

### Inspector UI active work items
- [ ] Fix `net_value_diff` display multiplier — currently ×20, should be ×50 to match minimax divisor and training scale
- [ ] Copy node button — tiny icon per row, copies path from root to that node only
- [ ] Condensed state copy format — compact and human-readable for pasting into prompts
- [ ] Load game state from plain text — paste a copied state string to jump to that position
- [ ] Node IDs — falls out naturally from move-path key scheme (see 8h)
- [ ] Scale indicator — tooltip clarifying all point values are `earned` differential ÷ 50, displayed ×50

---

## Multi-head Value Network

`AzulNet.forward(spatial, flat)` returns `(logits, value_win, value_diff, value_abs)`.

When adding callsites that consume net output, unpack or ignore auxiliary heads:
```python
logits, value_win, _value_diff, _value_abs = net(spatial, flat)
```

---

## Checkpoint Management

- `checkpoints/latest.pt` — current training weights; written every iteration and on promotion
- `checkpoints/latest_params.json` — args used to produce `latest.pt`
- `checkpoints/gen_xxxx.pt` — promoted checkpoints only
- `--load` defaults to `checkpoints/latest.pt`; silently skips if file does not exist
- `--initial-generation N` — start generation counter at N

To manually promote Phase 1 checkpoint before Phase 2:
```
copy checkpoints\latest.pt checkpoints\gen_0001.pt
```
Then run Phase 2 with `--initial-generation 1`.

---

## Hard-Won Lessons (Do Not Repeat)

1. **`value_only_iterations > 0` is a divergence trap.** Policy stays random, value predicts garbage. Always `--value-only-iterations 0`.
2. **One-hot policy targets poison the policy head.** Fix: `policy_distribution()` returns soft distributions.
3. **Random agent pretrain is nearly useless.** Floor-heavy games, near-zero score signal.
4. **Clearing buffer after pretrain kills value signal.** Never use `--clear-buffer-after-pretrain`.
5. **Eval at low sim counts is nearly useless.** Need ≥1500 sims for meaningful eval.
6. **Net weights reset every iteration when eval was skipped.** Fix: only reset when eval actually ran and net lost.
7. **Policy loss dominates the trunk (~50x value loss).** Use `--diff-only` Phase 1 so trunk develops value-relevant features first.
8. **Value head overfit to factory configurations.** Fix: encoding v3 removes raw factory distribution entirely.
9. **MCTS snowballs on high-value outliers.** Fix: value head calibration (Phase 1), not PUCT change.
10. **Eval against random `gen_0000.pt` produces move-cap hits.** Fix: copy `latest.pt` to `gen_0001.pt` after Phase 1, use `--initial-generation 1`.
11. **`uvicorn --reload` restarts server on checkpoint writes.** Run without `--reload` during training.
12. **`_ensure_expanded` called policy_value_fn twice per node.** Fix: `_ensure_expanded` returns the value; `_evaluate` reuses it.
13. **AlphaBeta as inspector prior is slow.** Every MCTS node expansion runs a full alpha-beta search. Expected behavior — not a bug. Ludacris depths will be very slow.
14. **`Move` is not hashable.** Cannot be used as dict key. Match by equality using `next()` over a list.
15. **Virtual loss concentration bug.** All root children get one visit; all remaining simulations pile onto one branch. Likely cause: virtual loss magnitude too small, or batch too large relative to sim count. Under active investigation.
16. **Virtual loss needs `visits + virtual_loss` as parent count in PUCT.** Passing only `visits` makes U=0 during batch collection (backprop hasn't run yet), collapsing exploration to prior-greedy selection within each batch. The highest-prior child accumulates all remaining selections after the first pass through all children.
---

## Deferred / Future Ideas

- **Parallel self-play via multiprocessing** — heuristic collection is parallel; self-play still sequential.
- **Encoding cache / state reuse** — within a single search, cache `(encoded_state, legal_moves)` per node. When a move is played, the child node becomes root with its children already encoded. Measure before implementing — may not be the bottleneck.
- **Shared game-owned state tree** — game owns a cache of encoded states and legal moves; agents request batches directly. Reduces re-encoding across agents in eval.
- **Node-budget AlphaBeta** — cap total nodes explored rather than depth. More principled than depth-based adaptive search. Deferred — depth-based is working well.
- **AlphaBeta depth selection via source count** — `sum(sources_with_color)` from flat encoding gives the distinct pair count. Deferred — legal move count is good enough for now.
- **Prior round board state snapshots** — one encoded state per completed round, up to 5. Try if model fails to generalize across rounds.
- **Elo UI page** — display Elo ladder across all `gen_xxxx.pt` from training logs.

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/train.py` | AlphaZero self-play training loop |
| `scripts/inspect_policy.py` | Per-move policy/value/MCTS diagnostic; encoding verification |
| `scripts/sample_policy.py` | Bulk value calibration across N random game states |
| `scripts/tournament.py` | Round-robin parallel tournament; games feed replay buffer |
| `scripts/benchmark_agents.py` | First-move vs overall timing by depth config |
| `scripts/benchmark_mcts.py` | MCTS throughput benchmarking |
| `scripts/self_play.py` | Generate self-play games |
| `scripts/parse_log.py` | Parse training logs |
| `scripts/migrate_recordings.py` | Migrate old recording format |
| `scripts/debug_minimax.py` | One-off diagnostic (virtual loss investigation) |

```
# Sample policy
python -m scripts.sample_policy --checkpoint checkpoints/latest.pt --samples 100 --summary-only
python -m scripts.sample_policy --checkpoint checkpoints/latest.pt --samples 20 --turn 5

# Inspect policy
python -m scripts.inspect_policy --checkpoint checkpoints/latest.pt --moves 3 --depth 1 --mcts-sims 200

# Tournament
python -m scripts.tournament --agents greedy minimax alphabeta_hard --games 200 --workers 8
```

---

## Change Log

| Date | Change |
|---|---|
| 2026-04-25 | Inspector agent selector, policy priors, visit fractions. Minimax value fixed. Double policy_value_fn call eliminated. Copy Tree includes agent header. Inspector UI todos logged. |
| 2026-05-02 | tests/engine/test_player.py added — 100% public interface coverage. Test subfolder structure introduced. |
| 2026-05-02 | Project plan and instructions overhauled. Encoding v3 designed. Engine rewrite (Player owns scoring) drafted. Virtual loss bug identified as top priority. Inspector serialization redesign fully specified. |
| 2026-05-03 | Virtual loss MCTS concentration bug fixed. Root cause: `node.visits` (always 0 during batch collection) passed as parent_visits to puct_score, collapsing exploration term to 0. Fix: pass `node.visits + node.virtual_loss` instead. |
| 2026-05-03 | `inspect_policy.py` probe fixed: removed stale v1 encoder imports from `_print_encoding`, deleted appended debug script, added `--batch-size` arg (default 8). |

> For full history, see `git log`.