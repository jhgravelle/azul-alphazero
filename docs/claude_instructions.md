# Claude Instructions — Azul AlphaZero Project

---

## Project context

I am building an Azul board game engine with an AlphaZero AI in Python. This is a learning project — I am a beginner-to-intermediate Python developer. Please explain concepts as you go and don't skip steps.

**Tech stack:** Python 3.12, pytest, black, flake8, FastAPI + HTML/JS, PyTorch, Git + GitHub, GitHub Actions, Windows 11

**Project structure and history:** See `PROJECT_PLAN.md` in the repo root.

---

## Current phase

> **Phase 8c — Training Run + Iteration**

- Strong heuristic baseline established: AlphaBetaAgent >> MinimaxAgent >> Greedy
- AlphaBeta hard (depths=3,5,8 thresholds=20,10) wins 76% vs minimax — this is now the promotion bar for AlphaZero
- Inspector UI complete with immediate/cumulative scores, start/stop, copy state/tree
- Ready to resume training with AlphaBeta as the evaluation target

---

## How I want to work

- **Shorter prompts:** We have Claude usage limits, don't endlessly run away, if you get stuck say so, and we can determine where to go next, or craft a prompt for another model to try.
- **TDD always:** You provide the test and implementation at the same time.
- **One step at a time:** Don't give me five files at once. Walk me through each piece.
- **Explain the why:** If you make a design decision, tell me why, especially if there's a tradeoff.
- **Git commits:** Tell me the commit message when it's a natural stopping point — after a feature is complete, a bug is fixed, or a refactor is done. Don't remind me after every single edit; I'll commit when the message makes sense as a unit of work.
- **Check CI:** We can proceed immediately after push but CI might go red and we have to backtrack to fix.
- **Repeat on request:** If I ask for something you've already provided, just repeat it without comment.
- **Sketch before coding:** For layout or visual changes, describe or sketch in text first to confirm we agree before writing any code.
- **Branch for larger refactors:** Use feature branches (`feat/xxx`) and draft PRs for work that spans multiple commits. CI only runs on `main` and PRs, so opening a draft PR is how to get CI on a feature branch.

---

## Code style preferences

- **f-strings for logging and string formatting.** I prefer `f"something: {value:.4f}"` over `"something: %.4f" % value` or `"something: {:.4f}".format(value)`. Keep this consistent in new code. Existing `%`-style logging calls can stay as-is unless I'm already editing them.
- **Always provide full methods:** when editing a method always provide the new complete method.
- **Break long methods into sub-methods.** When a function grows past ~20 lines or has multiple distinct phases, split it into named helpers. Prefer many small well-named functions over long ones with inline comments marking "sections."
- **Descriptive helper names over inline complexity.** A helper named `_apply_warmup_floor_override(move, policy_pairs, game)` is more readable than a 15-line inline block with a comment.
- **Avoid abbreviations.** Well named variables self document the code, short variable names only when immediately obvious to a beginner programmer and extremely small in scope.
- **Provide indented code.** When providing code it helps if it is indented to the appropriate level for easier copy/paste into the IDE.

---

## Critical gotchas

These have caused bugs before — always keep them in mind:

- `Move` uses `.tile`, not `.color`
- Always import `Tile` from `engine.constants`, never from `engine.tile`
- `_is_valid_destination` checks the specific wall column for that color in that row — not whether the color appears anywhere in the row
- `_score_floor` must filter out `Tile.FIRST_PLAYER` before adding to discard
- Empty `legal_moves()` mid-round is always a bug, never a valid game state
- `score_placement` precondition: tile must already be placed in the wall before calling
- **The API owns round transitions** — `_handle_round_end()` must be called after every `make_move` in the API; it calls `advance()` unconditionally
- **Simulation loops** (MCTS, AlphaZero, trainer) must call `game.advance()` after every `make_move`
- **Minimax/AlphaBeta** must call `advance(skip_setup=True)` and compute `earned_score_unclamped` BEFORE advance, not after — advance mutates the wall
- **Recordings:** when adding fields to `GameRecord` or `RoundRecord`/`MoveRecord`, also update `scripts/migrate_recordings.py`
- **Windows:** use `python -m module.path` to run scripts, set `$env:PYTHONPATH = "."` if needed. Use `findstr` not `grep`. No Unicode in log strings.
- **Test file name is `test_agent.py` (singular).** Not `test_agents.py`.
- **Adding a new agent** requires: `agents/registry.py` entry, `api/schemas.py` PlayerType update.

---

## Training/ML lessons learned (Phase 8c)

Things we've burned hours on; don't repeat these:

- **Value-only training for many iterations is a divergence trap.** The value head learns to accurately predict garbage outcomes while the policy head stays random. Self-play games get progressively worse. Use `--value-only-iterations 0` and train policy+value from iter 1.
- **One-hot policy targets from heuristic agents poison the policy head.** Fix: each agent exposes `policy_distribution(game)` returning the soft distribution it samples from. Heuristic pretrain pushes these distributions as policy targets, not one-hots.
- **Heuristic pretraining pairing matters.** Greedy-vs-Random: wide value-head signal but one-hot policy problem. Greedy-vs-Cautious: clean soft policy distributions, narrower value-head variance. Current default is Greedy-vs-Cautious. If `value_abs` goes dead, consider mixing Random back in.
- **Clearing buffer after pretrain kills `value_abs`.** Self-play data alone has near-uniform-bad scores early. Mixed buffer keeps the head alive. Don't use `--clear-buffer-after-pretrain` by default.
- **`_MAX_MOVES = 100` is the right round cap.** Human games max at ~65 moves; anything longer is pathological.
- **Per-head loss logging is essential.** `avg loss: X | policy: Y | value: Z (win A, diff B, abs C)` format.
- **Eval at low sim counts (100-200) is nearly useless.** Need 1500+ sims for meaningful eval, or a shared-state-tree redesign.
- **Eval games feed the replay buffer.** This is intentional — more data diversity than self-play alone.
- **GPU is not the bottleneck.** MCTS inference runs on CPU deliberately. Parallelism via `multiprocessing` should give near-linear speedup; `threading` won't due to the GIL.
- **AlphaZero promotion bar is now AlphaBeta hard**, not just Greedy. A checkpoint must beat `alphabeta_hard` (depths=3,5,8 thresholds=20,10) to be deployable.

---

## Training run configuration reference

Current known-good command shape:

```
python -m scripts.train \
  --iterations N \
  --games-per-iter G \
  --simulations S \
  --train-steps T \
  --pretrain-games PG \
  --pretrain-steps PS \
  --value-only-iterations 0 \
  --skip-eval-iterations 3 \
  --eval-games EG \
  --eval-simulations ES \
  --win-threshold 0.55
```

Sensible scale points:
- **Smoke test:** `N=2-3, G=5, S=200, T=200, PG=20, PS=300` — runs in ~2-3 min
- **Medium run:** `N=10, G=15, S=750, T=300, PG=100, PS=1500, ES=1500` — ~5-6 hours
- **Long run:** once stable, scale up iterations and games-per-iter.

Key guidance:
- `eval-simulations` should be `>=` self-play simulations.
- `skip-eval-iterations 3` lets self-play build a training buffer before eval starts.

---

## Multi-head value network notes

`AzulNet.forward(spatial, flat)` returns a 4-tuple: `(logits, value_win, value_diff, value_abs)`.

- `value_win` is the primary target, used by PUCT during search
- `value_diff` and `value_abs` are training-only auxiliaries
- All three share the trunk; each has its own Linear -> ReLU -> Linear -> Tanh head
- Loss: `policy + value_win + 0.3*value_diff + 0.3*value_abs`
- `ReplayBuffer.push()` and `.sample()` carry three value targets per example
- Target functions in `neural/trainer.py`: `win_loss_value`, `score_differential_value`, `total_score_value`

When adding new callsites that consume net output, unpack or ignore the auxiliary heads:
```python
logits, value_win, _value_diff, _value_abs = net(spatial, flat)
```

---

## Policy distribution system notes

All agents have a `policy_distribution(game) -> list[tuple[Move, float]]` method returning what they sample from.

- `Agent` base class: uniform over legal moves (default)
- `RandomAgent`: inherits default (behavior matches)
- `CautiousAgent`: uniform over non-floor moves
- `EfficientAgent`: uniform over partial-line moves (fallback to all)
- `GreedyAgent`: color-conditional — pick color uniformly, then uniform over preferred/all within color
- `MinimaxAgent` / `AlphaBetaAgent`: uniform (inherited) — these agents don't use policy_distribution for play

Used in `collect_heuristic_games` to produce soft policy targets (not one-hot) for pretrain.

When adding a new heuristic agent, override `policy_distribution` so pretrain gets soft targets instead of one-hots.

---

## Minimax/AlphaBeta agent notes

Both agents search to the round boundary naturally — depth limit is rarely reached because the round ends first. The adaptive depth system via `depths/thresholds` tuples controls how deep to search based on legal move count:

```python
MinimaxAgent(depths=(2, 3, 4), thresholds=(10, 7))
# >10 legal moves -> depth 2 (early round, high branching ~90 moves)
# 7-10 legal moves -> depth 3
# <=7 legal moves  -> depth 4
```

Key lesson: **compute `earned_score_unclamped` BEFORE calling `advance(skip_setup=True)`**. After advance, the wall is scored and pattern lines cleared — computing earned score after gives wrong results.

AlphaBeta move ordering uses a cheap heuristic (no cloning):
- Floor moves: bad for maximizer, good for minimizer
- Line-completing moves: good for maximizer, bad for minimizer
- Partial fills: neutral

Clone-based `_immediate_score` ordering was 16x slower and must never be reintroduced.

---

## Agent registry notes

`agents/registry.py` is the single source of truth. Structure:

```python
AGENT_REGISTRY: list[tuple[str, str, Callable, bool]] = [
    ("name", "Label", lambda **kwargs: AgentClass(**kwargs), hidden_bool),
    ...
]
```

`GET /agents` serves visible agents to the frontend. `make_agent(name, **kwargs)` constructs agents. Adding a new agent: add to registry, add to `PlayerType` in `schemas.py`.

UI difficulty levels (AlphaBeta):
- `alphabeta_easy`: `depths=(2,3,7), thresholds=(20,10)`
- `alphabeta_medium`: `depths=(3,5,7), thresholds=(20,10)`
- `alphabeta_hard`: `depths=(3,5,8), thresholds=(20,10)` — promotion bar for AlphaZero

---

## Inspector UI notes

The search tree inspector connects to `POST /inspect/live` (live game) or streams via the manual polling loop. Key behaviors:

- `is_stable()` returns True only when `_root._fully_explored` is True
- PUCT returns `-inf` for fully-explored nodes so selection skips them
- `_is_subtree_explored` propagates `_explored` bottom-up after each batch
- Transposition node reuse is disabled — nodes are not shared between parents
- `sim_count` in snapshot = `root.visits` (accurate)
- Immediate score = `earned_score_unclamped` delta, negated for opponent moves
- Cumulative = minimax rollup (maximize at root player's turns, minimize at opponent's)
- Children sorted by cumulative_immediate, alternating desc/asc by depth

---

## Branching workflow reminder

For work spanning multiple commits, use a feature branch (`feat/xxx`). Open a draft PR on GitHub so CI runs. Push commits to the branch as TDD progresses. Mark the PR ready when complete, merge, delete branch.

Common branching pitfalls:
- Forgetting to set upstream on first push — use `git push --set-upstream origin feat/xxx` or the Publish Branch button in VS Code
- CI not running on branches directly — only on `main` pushes and PR events. Opening a draft PR triggers CI