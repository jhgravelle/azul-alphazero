# project_plan.md

# Azul AlphaZero — Project Plan

> Last updated: 2026-05-07
> Status: Phase 8 in progress. 8x complete — training pipeline overhaul: ABeasy pretrain, ABeasy vs ABeasy buffer injection, dropout, showcase recordings, Game/_AlphaBetaAgent RNG fix, log_encoded_states AZ diagnostics.

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
True AlphaZero uses 800 simulations per move for both training and eval. Our count is TBD — Azul has a much smaller branching factor than chess/Go, so far fewer simulations may be sufficient. Default is 200. At 50 sims with 50+ legal moves, MCTS cannot differentiate moves — every move gets ~1 visit and the value head never accumulates Q-signal. 200 sims is the practical minimum for meaningful policy targets.

**Deviation 2 — Training game generation**
True AlphaZero generates training games from self-play only. We use a progressive curriculum: AZ vs AlphaBeta easy until 55% win rate, then switch to AZ vs AZ self-play. This prevents early self-play from poisoning the buffer with random-quality games. Each iteration also injects ABeasy vs ABeasy pairs (same count as AZ collection) to maintain high-quality policy signal independent of AZ's current strength.

**Deviation 3 — Evaluation approach**
True AlphaZero runs a fixed evaluation tournament with exploration off and promotes if win rate exceeds a threshold. We use the same approach: eval uses `temperature=0.0` (greedy/argmax), promotion at ≥55% win rate over 48 mirror pairs vs current best.

**Deviation 4 — Virtual loss and batching**
Our batched MCTS with virtual loss provides exploration breadth within each batch, partially compensating for any reduction in per-move simulation count. This is an architectural difference from the original sequential MCTS.

**Deviation 5 — Mirror games**
We use mirrored game pairs (identical bag seed, both players swap sides) for both training and evaluation. True AlphaZero does not do this — it relies on volume. Our mirror design controls for first-player advantage and the tile draw randomness within a game pair, providing stronger signal per game pair. The bag is only shuffled at game start and at refill; refill does not occur until round 6, which most high-quality games never reach. This guarantees identical factory contents for rounds 1–5 regardless of move choices.

**Deviation 6 — Pretrain phase**
Before the main iteration loop, optionally collect ABeasy vs ABeasy mirror pairs and train for many steps. This gives the policy head a warm start — it learns to suppress floor moves and prefer pattern line placement before AZ ever plays a game. Without pretrain, early AZ games produce nearly uniform MCTS visit distributions (50 sims / 50+ legal moves = ~1 visit each), which train the policy head on noise.

---

## Repository Structure

```
azul-alphazero/
├── agents/
│   ├── alphabeta.py       # Alpha-beta pruning, softmax policy_distribution, stochastic selection, per-instance RNG
│   ├── alphazero.py       # Thin wrapper — delegates to SearchTree
│   ├── base.py            # Agent base class + default policy_distribution (uniform)
│   ├── cautious.py        # Uniform over non-floor moves
│   ├── efficient.py       # Uniform over partial-line moves (fallback to all)
│   ├── greedy.py          # Color-conditional distribution
│   ├── mcts.py
│   ├── minimax.py         # Depth-limited minimax, source-adaptive depth
│   ├── move_filters.py    # non_floor_moves shared helper
│   ├── random.py          # Inherits uniform distribution
│   └── registry.py        # Single source of truth for all agents
├── api/
│   ├── main.py
│   └── schemas.py
├── engine/
│   ├── constants.py       # Tile enum, WALL_PATTERN, FLOOR_PENALTIES, all constants
│   ├── game.py            # Game controller: make_move, advance, legal_moves, tile_availability; instance RNG
│   ├── game_recorder.py   # GameRecorder, GameRecord, RoundRecord, MoveRecord
│   ├── player.py          # Player dataclass — owns all board state and scoring
│   └── replay.py          # replay_to_move
├── frontend/
│   ├── game.js
│   ├── index.html
│   ├── render.js
│   └── style.css
├── neural/
│   ├── encoder.py         # Flat MLP encoding (v3: 125-value vector, no spatial conv)
│   ├── model.py           # MLP-only architecture (1 ResBlock 64-dim, optional dropout, policy + 3× value heads)
│   ├── replay.py          # Circular replay buffer, three value targets + policy mask per example
│   ├── search_tree.py     # SearchTree: MCTS, background worker, subtree reuse
│   └── trainer.py         # compute_loss, Trainer, AgentSpec, parallel collection
├── scripts/
│   ├── benchmark_agents.py
│   ├── benchmark_mcts.py
│   ├── bench_score_placement.py
│   ├── debug_minimax.py   # One-off diagnostic (virtual loss investigation)
│   ├── inspect_policy.py  # Per-move policy/value/MCTS diagnostic tool
│   ├── log_encoded_states.py  # AZ vs AZ game log: encoding, AB targets, net predictions per turn
│   ├── migrate_recordings.py
│   ├── parse_log.py
│   ├── sample_policy.py   # Bulk value head calibration checker (N random states)
│   ├── self_play.py
│   ├── tournament.py      # Round-robin parallel tournament; games feed replay buffer
│   └── train.py
├── tests/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── test_alphabeta.py
│   │   └── test_minimax.py
│   ├── engine/
│   │   ├── test_game.py
│   │   └── test_player.py
│   └── [legacy flat test files — move to subfolders when touched]
├── checkpoints/           # gitignored; latest.pt = most recent training state
├── cli/
├── htmlcov/               # gitignored
└── recordings/            # gitignored; subfolders: training/, eval/, (root = human games)
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
- `Game.determine_winners() -> list[Player]` — returns list of one or two players. Tiebreak: (1) highest score after bonuses, (2) most completed horizontal wall rows, (3) shared victory. No column tiebreaker — not in official rules.
- Deleted: `engine/board.py`, `engine/game_state.py`, `engine/scoring.py`
- Deleted associated tests: `test_board.py`, `test_game_state.py`, `test_scoring.py`
- Moved `tests/test_game.py` → `tests/engine/test_game.py`

**Status:** Complete. 626 tests passing.

#### 8g — Virtual Loss / MCTS Bug Fix ✅

**Bug:** During a search, all root children get one visit each (correct). All remaining simulations then pile onto one branch despite virtual loss being applied.

**Root cause:** `_select_vl` passed `node.visits` as `parent_visits` to `puct_score`. During batch collection, backprop hasn't run yet, so `node.visits` is 0 for the root the entire batch. This makes the exploration term `U = C * prior * sqrt(0) / (1 + adj_visits) = 0` for every child. With U=0, PUCT collapses to Q-only selection. After all children have been visited once, the highest-prior child wins every subsequent selection in the batch.

**Fix:** In `_select_vl`, increment `virtual_loss` before computing `parent_visits`, then pass `node.visits + node.virtual_loss`.

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

Mirror game pairs (identical bag seed, sides swapped) used for all data collection. Forces value head to learn board evaluation rather than factory fingerprints. All collection functions in `trainer.py` use mirror pairs by default.

#### 8i — Encoding v3: Pure MLP ✅

Single flat vector (125 values) fed to MLP trunk + policy/value heads. No spatial branch.

**Flat vector layout:**

| Indices | Count | Content | Normalization |
|---|---|---|---|
| 0–24 | 25 | My pattern line fills (flattened 5×5) | Ratio: 0.0–1.0 per cell |
| 25–49 | 25 | My wall (flattened 5×5) | Binary: 0 or 1 |
| 50–74 | 25 | Opponent pattern line fills (flattened 5×5) | Ratio: 0.0–1.0 per cell |
| 75–99 | 25 | Opponent wall (flattened 5×5) | Binary: 0 or 1 |
| 100 | 1 | My official score | ÷ 100 |
| 101 | 1 | My earned score | ÷ 100 |
| 102 | 1 | Padding zero | — |
| 103 | 1 | Opponent official score | ÷ 100 |
| 104 | 1 | Opponent earned score | ÷ 100 |
| 105 | 1 | My floor penalty | ÷ 14 |
| 106 | 1 | I hold first-player token | Binary: 0 or 1 |
| 107 | 1 | Padding zero | — |
| 108 | 1 | Opponent floor penalty | ÷ 14 |
| 109 | 1 | Opponent holds first-player token | Binary: 0 or 1 |
| 110–114 | 5 | Tiles available by color | count ÷ 20 |
| 115–119 | 5 | Sources with that color | count ÷ 5 |
| 120–124 | 5 | Bag tile count by color | count ÷ 20 |

**Total: 125 values**

`_encode_flat_game_tiles` delegates to `game.tile_availability()` — encoder no longer reaches into `game.factories` or `game.center` directly.

#### 8r — Training v2: 3-Head Policy + Temperature + Mirror Eval + Pretrain ✅

**1. Three-head policy** — `policy_head` (flat 180) replaced by three independent heads:
- `source_head` (2): center vs factory
- `tile_head` (5): tile color
- `destination_head` (6): pattern line + floor

Move priors = softmax(src) × softmax(tile) × softmax(dst), renormalized over legal moves.
Training targets: flat MCTS visit distribution marginalized to per-head soft targets via
`flat_policy_to_3head_targets()` in `encoder.py`.

Note: `MOVE_SPACE_SIZE = NUM_SOURCES(7) × BOARD_SIZE(5) × NUM_DESTINATIONS(6) = 210`. Flat index = source_idx × 30 + color_idx × 6 + dest_idx.

**2. Temperature exploration (25-move rule)** — `_pick_move` reads `root.game.turn`:
- Moves 0–24: `EXPLORATION_TEMP = 1.0` (stochastic)
- Moves 25+: `DETERMINISTIC_TEMP = 0.1` (sharp)
- Eval always uses `temperature=0.0` (argmax)

**3. Value simplification** — `value_abs` excluded from training loss. Combined value loss:
`0.3×value_win_loss + 1.0×value_diff_loss`. `value_abs` still computed and logged.
MCTS uses `value_diff` as its Q signal — continuous, better gradient than binary `value_win`.

**Known limitation:** The three-head factorization assumes source, color, and destination are independent — they are not. Once a destination row is chosen, the color is nearly determined (each row holds one color). A high src×tile product for a floor move can overcome a low dst probability, causing floor moves even when the destination head correctly assigns floor low probability. A single destination-only head may be better.

**Breaking change:** checkpoint format incompatible with pre-8r checkpoints (policy head renamed).

#### 8s — Parallel Training Pipeline + AZ vs AB Progression ✅

**Unified AgentSpec worker architecture:**
- `AgentSpec` dataclass — serializable description of any agent (AlphaZero or heuristic). Picklable, passed to worker processes.
- `_build_agent(spec)` — constructs any agent from a spec inside a worker process
- `_worker_play_mirror_pair(spec_0, spec_1)` — single worker function; picks its own seed, plays game A + game B (sides swapped), returns serializable records
- `_iter_pair_results(sampled, num_workers)` — generator using `imap_unordered`; yields results as workers finish for streaming log output. `KeyboardInterrupt` terminates workers cleanly.
- `collect_parallel(buf, spec_0, spec_1, num_pairs, num_workers)` — fixed matchup mirror pairs
- `collect_heuristic_parallel(buf, num_pairs, matchups, num_workers)` — weighted matchup sampling
- `collect_ab_parallel(buf, num_pairs, num_workers)` — ABeasy vs ABeasy injection
- `evaluate_parallel(new_net, old_net, num_pairs, simulations, buf, num_workers)` — eval with `temperature=0.0`, runs all pairs to completion, pushes to buf only when in az-vs-az mode

**Pair logging:** demoted from INFO to DEBUG — only summary lines remain at INFO.

**AZ vs ABeasy progression:**
- Iterations start in `az-vs-abeasy` mode
- Each iteration: collect N AZ vs ABeasy pairs + N ABeasy vs ABeasy pairs; check win rate from AZ collection
- If AZ win rate ≥ 55%: switch to `az-vs-az` mode next iteration (permanent)
- Both modes: follow with training steps, then eval pairs new vs best net for promotion
- **Eval games NOT pushed to buffer while in az-vs-abeasy mode**

**Pretrain (optional, single pass before iteration loop):**
- `--pretrain` flag: collect `buffer_size // 100` ABeasy vs ABeasy pairs, train for `num_pairs × 20` steps
- Saves `gen_0000.pt` — pretrained baseline, not random weights
- `best_net` initialized from pretrained weights so iteration 1 evals against gen_0000
- Purpose: give policy head a warm start so MCTS has a meaningful floor-suppressing prior from iteration 1

**Default CLI args:**
```
--workers 8  --simulations 200  --games-per-iter 200  --eval-games 48  --train-steps 10000  --dropout 0.1
```

#### 8t — Training Fixes & Model Shrink ✅

**Bug fix — `_compute_game_scores` double-counted end-of-game bonuses.**
After `handle_game_end()` runs, `player.bonus` is added into `player.score` but `bonus` itself is not zeroed. Reading `player.earned` (= `score + pending + penalty + bonus`) after game end double-counts the bonus. Fix: read `player.score` at game end — it is the canonical settled value.

**Model shrink:** `AzulNet` reduced from 256-dim (~215k params) to 64-dim (~8k params). No intermediate layer in value heads. Checkpoint format incompatible with pre-8t checkpoints.

**Training steps:** increased from 500 to 10,000 per iteration. Training steps are cheap (faster than a single eval game) so running more steps per iteration is effectively free.

**AlphaZero added to inspector UI:** `agents/registry.py` and `api/schemas.py` updated. `_load_az_net()` loads `checkpoints/latest.pt` at agent construction time — picks up latest weights when server restarts during training.

#### 8u — Training Signal Fixes ✅

**Bug fix — `_terminal_value` divisor was ÷20, should be ÷50.**
`SearchTree._terminal_value` returned `(my_earned - opp_earned) / 20.0`, but the `value_diff` head is trained with `_SCORE_DIFF_DIVISOR = 50.0`. Terminal node Q-values were 2.5× larger than the network's trained output range, corrupting MCTS policy distributions and creating a negative feedback loop: biased policy targets → net learns to prefer fast game endings → faster losses against AB medium → more negative buffer data → net regresses. Fix: import and use `_SCORE_DIFF_DIVISOR` from `trainer.py` in `_terminal_value`.

**Round-boundary states added to training buffer.**
MCTS evaluates round-boundary leaf nodes (empty factories, committed pattern lines, pending scores unsettled) using the neural net. These states never appeared in the training buffer — the net was extrapolating to out-of-distribution inputs at every round boundary. This matters strategically: the best move at the end of a round is sometimes to trash a tile to the floor (small immediate penalty) to keep a pattern line clean for next round. A heuristic terminal value would favor blocking the line (no penalty), so using the net is essential. Fix: after each `make_move` that ends a round, `_play_game` clones the game, calls `next_player()` (matching MCTS child construction exactly), encodes the boundary state, and appends it to history with `policy_valid=False`.

**Policy mask added to replay buffer and loss computation.**
Round-boundary examples have no legal moves and no policy target. `ReplayBuffer` gains a `policy_masks` field (1.0 = train policy, 0.0 = value-only). `compute_loss` masks out boundary examples from the policy loss and averages over valid examples only. Value heads train on all examples including boundaries.

#### 8v — Code Review & Bug Fixes ✅

**mcts.py:**
- **Bug fix — `_backpropagate` did not alternate perspective.** Values were accumulated from a fixed root player's perspective instead of being negated at each level. Now uses `result = -result` per level (±1 convention), matching `search_tree.py`.
- **Bug fix — `_simulate` called `legal_moves()` twice per loop iteration.** Result cached in `legal` variable.
- **Value convention unified.** Both `mcts.py` and `search_tree.py` now use ±1 (positive = good for current player at that node), negated at each backpropagation level.
- Removed unused `import logging` / `logger`.
- Added `MCTSNode.is_terminal()`, `path_from_root()`, `_replay_to_node()` helpers.
- `choose_move` now raises `RuntimeError` on no legal moves; falls back gracefully if `simulations=0`.

**search_tree.py:**
- **Bug fix — `_empty_node_dict` schema was incomplete.** Missing `immediate`, `cumulative_immediate`, `minimax_value`, `net_value_diff`, `visit_fraction` keys.
- Refactored large methods into named helpers. Abbreviated names expanded. Full docstrings added.

**neural/replay.py:**
- **Bug fix — `sample` return type annotation declared 5 tensors, returns 6.** Added `policy_masks`.

**neural/trainer.py:**
- **Bug fix — Windows KeyboardInterrupt flood.** Workers install `signal.SIG_IGN` via `_worker_ignore_sigint` pool initializer.

**Tests updated:** `test_trainer.py`, `test_train.py`, `test_model.py` — all stale imports and API mismatches fixed.

#### 8w — AlphaBeta/Minimax Redesign + Training Buffer Fix ✅

**Source-adaptive depth for AlphaBeta and Minimax.**
Both agents now use `(depth, threshold)` instead of `(depths, thresholds)`. `_effective_depth` calls `game.tile_availability()` and sums source counts. When sources > threshold, uses fixed `depth`. When sources ≤ threshold, uses `sources` as depth — searching as deep as moves remain this round.

**Stochastic move selection for AlphaBeta.**
`AlphaBetaAgent` samples from its scored move distribution rather than always picking the maximum. Two temperature modes: normal play `exploration_temperature=0.3`, end-of-round `_END_OF_ROUND_TEMPERATURE=1.0`.

**UI difficulty presets updated:**

| Name | depth | threshold |
|---|---|---|
| `alphabeta_easy` | 1 | 4 |
| `alphabeta_medium` | 2 | 6 |
| `alphabeta_hard` | 3 | 8 |
| `alphabeta_extreme` | 4 | 10 |

`alphabeta_ludacris` removed. `AgentSpec` updated: `depths`/`thresholds` → `depth`/`threshold`.

#### 8x — Training Pipeline Overhaul + Diagnostics ✅

**Game RNG fix.**
`Game.__init__` replaces `random.seed()` + `random.shuffle()` with `self._rng = random.Random(seed)`. `_refill_bag` uses `self._rng.shuffle()`. `clone()` copies `_rng` state via `getstate()`/`setstate()`. This eliminates global random state pollution that caused AB vs AB mirror pairs to produce identical scores in both games — the two games in a mirror pair now diverge as intended.

**AlphaBetaAgent RNG fix.**
Per-instance `self._rng = random.Random()` for stochastic move selection. Independent from game RNG and other agent instances.

**AzulNet dropout.**
Optional `dropout: float = 0.0` parameter on `AzulNet`. `nn.Dropout(p=dropout)` applied to trunk output after ResBlock, before all heads. Active during `net.train()`, passthrough during `net.eval()`. Default in training: `--dropout 0.1`.

**Pretrain on ABeasy vs ABeasy.**
`_pretrain_matchups()` updated to ABeasy vs ABeasy (was Greedy vs ABeasy). Pretrain now fills the buffer with high-quality games that suppress floor moves. Policy head learns "don't floor" before AZ ever plays.

**ABeasy vs ABeasy buffer injection.**
`collect_ab_parallel(buf, num_pairs, num_workers)` added to `trainer.py`. Called each iteration during az-vs-abeasy mode with the same `games_per_iter` count as AZ collection. Every iteration gets `2 × games_per_iter` new examples: half from AZ play, half from ABeasy play. Maintains strong policy signal regardless of AZ's current strength.

**Removed:** `--heuristic-pairs-per-iter`, `--skip-eval-iterations` CLI args. `collect_heuristic_parallel` no longer imported in `train.py`.

**Showcase recordings.**
`_record_showcase_game(net, simulations, iteration, device)` plays one greedy AZ vs AZ game after each training phase and saves to `recordings/training/iter_NNNN.json`. Player names are `Iter N` vs `Iter N`. Not pushed to buffer — diagnostic only. Net moved to CPU for inference, then back to CUDA.

**API recordings.**
`list_recordings`, `get_recording`, and `_inspector_load` all scan `recordings/training/` in addition to `recordings/` and `recordings/eval/`. Frontend groups training recordings under "Training Games" section.

**log_encoded_states.py.**
Rewritten to run AZ vs AZ (or AB vs AB). Shows per-turn: encoded state grid, AB policy targets with marginals, net forward pass predictions (src/tile/dst head probabilities + value heads). Chosen move logged in turn header. Parameterizable `--top-n` for legal move display (default 6).

**Simulation count finding.**
At 50 sims with 50+ legal moves, each move gets ~1 visit — MCTS cannot differentiate and value head Q-signal never accumulates. Floor suppression requires ≥200 sims for the value head to act as a meaningful tiebreaker. Default raised to 200.

#### 8k — Elo Ladder

- Elo tracked live during training, logged per iteration
- Tournament games double as training data
- Goal: UI page showing Elo ladder across all `gen_xxxx.pt` checkpoints

#### 8l — Evaluation & Variance Analysis

**Goal:** Once a strong model exists, quantify how often the stronger player wins to understand Azul's luck/variance profile.

#### 8m — Parallel Eval Games ✅

Implemented via `evaluate_parallel` in `trainer.py`. Uses same `_iter_pair_results` / `imap_unordered` architecture as collection. All pairs run to completion, no early exit.

#### 8n — Multi-Flavor Hyperparameter Search (Early Stage)

**Goal:** Run multiple model flavors with differing hyperparameters in parallel during early training to identify the best model structure before committing to a full training run.

#### 8o — Training Dashboard & Remote Monitoring

**Goal:** Web dashboard for monitoring training progress, accessible remotely.

**Metrics:** Loss curves, Elo progression, win rates, buffer size, games per hour.

#### 8y — Single Destination Policy Head (Planned)

**Motivation:** The three-head factorized policy (src × tile × dst) has a structural flaw: a high src×tile product for a floor move can overcome a low dst probability, causing floor moves even when the destination head correctly suppresses them. This was observed in production — the pretrained net showed `0.05 Floor` in dst predictions but AZ still played floor moves at 50 sims.

**Proposed design:** Replace the three heads with a single `destination_head` (6 outputs: 5 pattern lines + floor). Prior for a legal move = `softmax(dst)[dest_idx]`, renormalized over legal moves. Training target = `dst_targets` already computed by `flat_policy_to_3head_targets`. Color and source selection would be handled downstream (pick the legal move with the chosen destination that maximizes some secondary criterion, or just pick the first legal move for that destination).

**Tradeoff:** Loses expressiveness when multiple colors are available for the same destination row. Gains: cleaner floor suppression, simpler loss, fewer parameters.

---

### Phase 9 — Polish and Release (wishlist)

- Animated tile placement
- Sound effects
- Cloud deployment
- README with screenshots
- Capacitor iOS/Android packaging (low priority)

---

## Backlog

- [ ] Write mirror game invariant test — confirm identical factory contents rounds 1–5 with same bag seed using random agents; assert round 6 is first potential divergence point.
- [x] Fix `_terminal_value` divisor — was ÷20, now ÷50 via `_SCORE_DIFF_DIVISOR` (8u)
- [ ] Fix `net_value_diff` display multiplier in inspector UI — label still shows ×20, should be ×50
- [ ] Copy node button — tiny icon per row, copies path from root to that node only
- [ ] Condensed state copy format — compact and human-readable for pasting into prompts
- [ ] Load game state from plain text — paste a copied state string to jump to that position
- [ ] Scale indicator — tooltip clarifying all point values are `earned` differential ÷ 50, displayed ×50
- [ ] Inspector serialization redesign (8h) — deferred in favour of training pipeline work
- [ ] Delete `neural/zobrist.py` — replaced by move-path node keys (see 8h)
- [ ] Single destination policy head (8y) — replace 3-head factorized policy with destination-only head

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
tile_availability()          — {color: (total_tiles, source_count)} across all sources
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

At game end, read `player.score` (not `player.earned`) — `bonus` has been folded into `score` by `handle_game_end()` and reading `earned` would double-count it.

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

### AlphaBeta and Minimax depth configuration

```python
AlphaBetaAgent(depth=d, threshold=t, exploration_temperature=e)
MinimaxAgent(depth=d, threshold=t)
# sources > threshold  -> use fixed depth d
# sources <= threshold -> use sources as depth (searches full remaining round)
# sources = sum of source_count values from game.tile_availability()
```

UI difficulty levels:
- `alphabeta_easy`: `depth=1, threshold=4`
- `alphabeta_medium`: `depth=2, threshold=6`
- `alphabeta_hard`: `depth=3, threshold=8` — promotion bar
- `alphabeta_extreme`: `depth=4, threshold=10`

---

## Inspector UI Reference

### Current capabilities
- Agent selector: Minimax, AlphaBeta Easy/Medium/Hard/Extreme, AlphaZero
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
- Recordings grouped by folder: Training Games / Eval Games / Human Games

### Minimax value semantics
`(current_player.earned - opponent.earned) / 50.0` at every node.

### Inspector UI active work items
- [ ] Fix `net_value_diff` display multiplier in inspector UI — label still shows ×20, should be ×50
- [ ] Copy node button — tiny icon per row, copies path from root to that node only
- [ ] Condensed state copy format — compact and human-readable
- [ ] Load game state from plain text
- [ ] Node IDs — falls out naturally from move-path key scheme (see 8h)
- [ ] Scale indicator — tooltip clarifying all point values are `earned` differential ÷ 50

---

## Multi-head Value Network

`AzulNet.forward(encoding)` returns `((src_logits, tile_logits, dst_logits), value_win, value_diff, value_abs)`.

```python
(src_logits, tile_logits, dst_logits), value_win, value_diff, value_abs = net(encoding)
```

Policy is factored into three independent heads (center vs factory, tile color, destination). Priors
per legal move = softmax(src)[src_type] × softmax(tile)[t] × softmax(dst)[d], renormalized.
Training targets marginalize flat MCTS visit distribution to per-head soft targets.

`value_abs` is computed and logged but excluded from the training loss. Loss formula:
`policy_loss + 0.3×value_win_loss + 1.0×value_diff_loss`.

MCTS uses `value_diff` as its Q signal. `value_diff` is continuous (score differential ÷ 50),
giving better gradient than the binary `value_win`.

Architecture: `input_proj(125→64) → 1×ResBlock(64) → Dropout(p) → 6 heads`. ~8k params.

---

## Checkpoint Management

- `checkpoints/latest.pt` — current training weights
- `checkpoints/latest_params.json` — args used to produce `latest.pt`
- `checkpoints/gen_xxxx.pt` — promoted checkpoints only
- `gen_0000.pt` — pretrained baseline (after `--pretrain`), not random weights
- `--load` defaults to `checkpoints/latest.pt`; silently skips if missing
- `--initial-generation N` — start generation counter at N
- When loading a specific gen checkpoint, always pass `--initial-generation N` to match

---

## Hard-Won Lessons (Do Not Repeat)

1. **`value_only_iterations > 0` is a divergence trap.** Always `--value-only-iterations 0`.
2. **One-hot policy targets poison the policy head.** Fix: `policy_distribution()` returns soft distributions.
3. **Random agent pretrain is nearly useless.** Floor-heavy games, near-zero score signal.
4. **Clearing buffer after pretrain kills value signal.** Never use `--clear-buffer-after-pretrain`.
5. **Eval at low sim counts is nearly useless.** Need ≥1500 sims for meaningful eval.
6. **Net weights reset every iteration when eval was skipped.** Fix: only reset when eval actually ran and net lost.
7. **Policy loss dominates the trunk (~50x value loss).** Use `--diff-only` Phase 1.
8. **Value head overfit to factory configurations.** Fix: encoding v3 removes raw factory distribution; mirror games eliminate factory-outcome correlation.
9. **MCTS snowballs on high-value outliers.** Fix: value head calibration (Phase 1).
10. **Eval against random `gen_0000.pt` produces move-cap hits.** Fix: copy `latest.pt` to `gen_0001.pt` after Phase 1.
11. **`uvicorn --reload` restarts server on checkpoint writes.** Run without `--reload` during training.
12. **`_ensure_expanded` called policy_value_fn twice per node.** Fix: `_ensure_expanded` returns the value; `_evaluate` reuses it.
13. **AlphaBeta as inspector prior is slow.** Expected behavior — not a bug.
14. **`Move` is not hashable.** Cannot be used as dict key. Match by equality using `next()` over a list.
15. **Virtual loss needs `visits + virtual_loss` as parent count in PUCT.** Passing only `visits` makes U=0 during batch collection, collapsing exploration to prior-greedy selection within each batch.
16. **`while not game.is_game_over()` as a loop condition is always wrong** — `is_game_over()` now requires both `has_triggered_game_end()` AND `is_round_over()`. Use `while not game.is_game_over()` only after confirming this is what you mean. In practice, game loops should check `is_game_over()` after `advance()`, not before `make_move()`.
17. **`has_triggered_game_end()` does not mean the game is over** — it means the game will end at the end of the current round. Moves may still remain this round. `is_game_over()` is only True when the round has also fully ended and all scoring is complete.
18. **Early self-play poisons the buffer.** A weak net generates near-random MCTS visit distributions. Self-play data displaces good pretrain data and loss increases. Fix: use AZ vs AlphaBeta easy until AZ reaches 55% win rate before switching to self-play.
19. **`imap_unordered` with multiprocessing needs a single-argument wrapper.** `pool.starmap` blocks until all results are done — use `imap_unordered` with a tuple-unpacking wrapper function to stream results as workers finish.
20. **Don't pickle agent specs back through the worker result queue.** Returning `AgentSpec` (which contains the full model state dict) from workers serializes weights on every result. Return a pair index instead and look up the spec in the main process.
21. **`_compute_game_scores` must read `player.score`, not `player.earned`, at game end.** After `handle_game_end()`, bonuses are folded into `score` but `bonus` is not zeroed — reading `earned` double-counts them. Scores of 0 in game logs are genuinely bad play (floor penalties exceeding placement points), not early termination.
22. **500 training steps per iteration is insufficient.** Training steps are essentially free relative to game collection and eval. Use 10,000+ steps per iteration to give the net meaningful gradient signal before evaluation.
23. **Pretrain loss curve not converged at 10k steps.** The curve was still declining — 50k steps reaches a better floor (~2.22 policy loss vs ~2.40 at 10k). Run pretrain longer.
24. **`_terminal_value` divisor must match `_SCORE_DIFF_DIVISOR`.** Using ÷20 while the value_diff head trains with ÷50 makes terminal Q-values 2.5× too large. MCTS policy distributions are biased toward fast game endings, generating corrupted training targets and a progressive regression loop. Both must use the same constant.
25. **Round-boundary states must be in the training buffer.** MCTS evaluates leaf nodes at round boundaries (empty factories, committed pattern lines). If the net has never seen these states during training, it extrapolates — and the heuristic fallback (`_terminal_value`) is wrong for Azul because it ignores the future cost of a blocked pattern line. Capture boundary states in `_play_game` after `make_move` (before `advance`), call `next_player()` to match MCTS child construction, and push with `policy_mask=0.0` so only the value heads train on them.
26. **`_backpropagate` in `mcts.py` must negate, not flip.** Using `1.0 - result` only works in the [0,1] range and is wrong conceptually — it conflates perspective-flipping with value transformation. Use `result = -result` (±1 convention) so both `mcts.py` and `search_tree.py` share the same mental model: positive = good for current player at that node.
27. **On Windows, Ctrl+C floods the terminal with worker KeyboardInterrupt tracebacks.** The OS broadcasts SIGINT to all processes in the console group. Fix: pass `initializer=_worker_ignore_sigint` to `ctx.Pool` — workers install `signal.SIG_IGN` on startup and let the main process handle shutdown via `pool.terminate()`.
28. **Stale tests are worse than no tests.** Tests importing functions that no longer exist (`collect_heuristic_games`, `evaluate`, `AzulNet(hidden_dim=...)`) block the entire test run. Delete or update them promptly — if the production API changed, the tests are wrong, not the code.
29. **Eval games poison the buffer during az-vs-abeasy phase.** Each iteration runs eval pairs of AZ vs AZ. Early in training these are low-quality and dilute the high-quality pretrain data. Fix: pass `buf=None` to `evaluate_parallel` while in az-vs-abeasy mode.
30. **AlphaBeta with `threshold=999` causes hangs.** High threshold means `sources > threshold` is almost never true, so `_effective_depth` falls through to `return sources`. Early round with 20+ source-colors = depth 20+. Use `threshold=0` to force fixed depth in tests.
31. **Game.__init__ resets global random state.** `random.seed(self.seed)` + `random.shuffle(self.bag)` in `__init__` resets the module-level `random` state, causing AB vs AB mirror pairs to produce identical scores (both games traverse the same random sequence). Fix: use `self._rng = random.Random(seed)` throughout — game RNG is fully instance-local.
32. **AlphaBetaAgent shares module-level random with the game.** Even after fixing Game RNG, two AB agents in the same process share `random.choices` state, producing identical stochastic decisions. Fix: `self._rng = random.Random()` per agent instance; pass to `_sample_from_distribution`.
33. **50 simulations is too few for meaningful MCTS policy targets.** With 50 sims and 50+ legal moves, each move gets ~1 visit. PUCT collapses to prior-only selection and the value head never accumulates Q-signal. Floor suppression requires the value head to act as a tiebreaker — which only works at ≥200 sims. Use 200 sims minimum for training collection.
34. **Three-head factorized policy can produce floor moves despite low floor dst probability.** The factored prior = softmax(src) × softmax(tile) × softmax(dst). A high src×tile product for a floor move can overcome a low dst[floor] value. This was observed with the pretrained net: dst predictions showed 0.05 Floor but AZ still played floor moves at 50 sims. A single destination-only head would be cleaner.
35. **`showcase_game` net must be moved to CPU before inference.** The showcase game runs in the main process where `net` lives on CUDA, but `AlphaZeroAgent` constructs its policy_value_fn expecting CPU tensors (workers are CPU-only). Call `net.cpu()` before constructing agents, then `net.to(device)` after saving the recording.

---

## Deferred / Future Ideas

- **Encoding cache / state reuse** — cache `(encoded_state, legal_moves)` per node within a search.
- **Shared game-owned state tree** — game owns a cache of encoded states; agents request batches directly.
- **Node-budget AlphaBeta** — cap total nodes explored rather than depth.
- **Prior round board state snapshots** — one encoded state per completed round, up to 5.
- **Single destination policy head** — replace 3-head factorized policy with destination-only head to cleanly suppress floor moves (see 8y).

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/train.py` | AlphaZero training loop — pretrain, AZ vs ABeasy, AZ vs AZ self-play |
| `scripts/log_encoded_states.py` | AZ vs AZ game log: encoding, AB targets, net predictions per turn |
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
| 2026-05-07 | **8x complete:** Game instance RNG (`self._rng`) fixes AB vs AB mirror pair determinism. AlphaBetaAgent per-instance RNG fixes stochastic move sharing. AzulNet gains optional dropout (default 0.1). Pretrain switched to ABeasy vs ABeasy. `collect_ab_parallel` injects ABeasy pairs every iteration. `--heuristic-pairs-per-iter` and `--skip-eval-iterations` removed. Showcase recordings saved to `recordings/training/` per iteration. API and frontend updated to scan training folder. Pair log lines demoted to DEBUG. Default sims raised to 200. `log_encoded_states.py` rewritten for AZ vs AZ with net predictions. |
| 2026-05-07 | **8w complete:** `Game.tile_availability()` added. AlphaBeta and Minimax redesigned: `depths`/`thresholds` → `depth`/`threshold`, source-adaptive depth. AlphaBeta gains stochastic move selection. UI presets updated. `AgentSpec` updated. Eval games no longer pushed to buffer during az-vs-abeasy phase. Tests moved to `tests/agents/`. 624 tests passing. |
| 2026-05-06 | **8v complete:** mcts.py backprop perspective bug fixed (±1 convention unified). `_simulate` double legal_moves call fixed. `search_tree.py` refactored. `_empty_node_dict` schema completed. `ReplayBuffer.sample` type annotation fixed. Windows KeyboardInterrupt flood fixed. Stale tests updated. |
| 2026-05-06 | **8u complete:** `_terminal_value` divisor fixed ÷20→÷50. Round-boundary states captured in training buffer. `ReplayBuffer` gains `policy_masks`. `compute_loss` masks boundary examples from policy loss. 545 tests passing. |
| 2026-05-06 | **8t complete:** `_compute_game_scores` bug fixed. `AzulNet` shrunk to 64-dim (~8k params). Training steps increased to 10k/iter. AlphaZero registered in inspector UI. |
| 2026-05-05 | **8s complete:** Unified AgentSpec worker architecture. `collect_parallel`, `collect_heuristic_parallel`, `evaluate_parallel` replace previous collection functions. AZ vs AB mode with auto-switch. Pretrain is single pass before loop. |
| 2026-05-05 | **8r complete:** 3-head policy (source×tile×dest), 25-move temperature rule, mirror eval, `--pretrain` mode. 626 tests passing. Checkpoint format incompatible with pre-8r. |
| 2026-05-05 | **8f cleanup complete:** Deleted engine/board.py, engine/scoring.py, engine/game_state.py. Moved tests/test_game.py → tests/engine/test_game.py. |
| 2026-05-05 | **Mirror games discovery:** Value head overfitting resolved by mirrored game pairs. |
| 2026-05-05 | **Encoding v3 (8i):** Pure MLP, flat 125-value vector. No spatial conv. All checkpoints incompatible. |
| 2026-05-04 | **MCTSAgent 3.5× speedup:** `clone()` replaces `deepcopy`, round-boundary rollouts, 200 sims default. |
| 2026-05-03 | **Engine rewrite (8f) complete** — Player owns all scoring, Game owns transitions. |
| 2026-05-03 | **Virtual loss MCTS concentration bug fixed.** Fix: pass `node.visits + node.virtual_loss`. |
| 2026-05-03 | **Fixed is_game_over()** — now requires is_round_over() AND has_triggered_game_end(). |

> For full history, see `git log`.