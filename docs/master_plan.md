# Azul Supervised Learning — Master Plan

**Last Updated:** 2026-05-15  
**Status:** Phase 0 in progress — Player encoding refactor  
**Branch:** `feat/supervised-value-net`

---

## Vision

Build a value network for the Azul board game using **supervised learning from AlphaBeta games**, not self-play. The approach:

1. Play strong AlphaBeta games (low temperature, high depth)
2. Collect every position labeled with the final game outcome
3. Train a supervised value network on (state, outcome) pairs
4. Improve AlphaBeta by replacing earned-score heuristic with learned value
5. Iterate: stronger net → better AlphaBeta → better training data

**Why this works:** Azul is shallow (~30 moves). AlphaBeta already plays well with a hand-coded heuristic. We bootstrap from that strength rather than rebuilding from self-play noise.

---

## Why This Pivot?

### AlphaZero Attempt Failed
- Trained for 8 phases with millions of self-play games
- Network never learned to suppress floor moves
- MCTS visits remained near-uniform despite value head
- No scaling: better nets produced worse policies

### Supervised Learning Succeeds
- [[alphabeta_strategy.md]] — AlphaBeta already beats most heuristics using earned score
- Deterministic games produce reliable training signal
- Value net improves AlphaBeta directly (no policy noise)
- Problem is shallow: 30 moves per game, supervised learning is fast

---

## Key Design Constraint: Round Boundaries

[[alphabeta_strategy.md#round_boundaries]] — Bot evaluation does not cross round boundaries.

**Why this matters:**
- Prevents expensive round-end scoring inside game tree search
- Forces value net to generalize end-of-round positions (don't memorize lucky next round)
- Keeps bot context small and tractable
- Matches how humans think: "end of round score, then next round begins"

---

## Current Phase: Phase 0 — Player Encoding Refactor

[[phase_0_refactor.md]] — Validate and refactor the 168-value player encoding before downstream work.

**Current state:**
- Player encoding implemented (25 lines of code per feature section)
- Tests written but not yet all passing
- Ready to refactor Game and external callers once tests pass

**Timeline:** ~17 hours total (3h tests, 4h Game refactor, 2h Game tests, 8h external callers)

---

## Encoding Strategy

[[encoding_strategy.md]] — Comprehensive feature design for the 168-value player encoding.

Covers:
- Wall state (25 values: binary cell states)
- Demand grids (50 values: per-color needs by row/col/total)
- Pattern lines (30 values: filled cells per row, available capacity)
- Scoring (5 values: official, pending, penalty, bonus, earned)
- Misc (3 values: first-player token, used tiles, max capacity)

Also documents **redundant features we've considered and rejected**.

---

## AlphaBeta Strategy

[[alphabeta_strategy.md]] — Why AlphaBeta + earned-score heuristic is the training target.

Covers:
- Why it's strong (76% vs Minimax, 55%+ vs self-play AZ)
- Why depth constraints work (source-adaptive depth, round boundaries)
- How it trains value nets iteratively
- Difficulty presets (easy/medium/hard/extreme)

---

## Next Phases (Planned)

### Phase 1 — Supervised Training Loop
- Generate games with AlphaBeta at varying depths
- Collect (state, outcome) pairs per game
- Train supervised value network
- Replace earned-score heuristic with learned value in AlphaBeta
- Evaluate: new-value-net AlphaBeta vs old-heuristic AlphaBeta

### Phase 2 — Iteration & Scaling
- Play stronger AlphaBeta (higher depths)
- Train more iterations
- Hyperparameter search (model width, dropout, learning rate)
- Track value net performance across generations

### Phase 3 — Analysis & Polish
- Elo ladder across generations
- Win rate analysis (variance/luck profile)
- UI improvements if needed
- Documentation and release

---

## Quick Reference

| File | Purpose |
|------|---------|
| [[master_plan.md]] | This file — vision and phase overview |
| [[alphabeta_strategy.md]] | Why AlphaBeta works, design constraints |
| [[encoding_strategy.md]] | 168-value encoding, feature rationale |
| [[phase_0_refactor.md]] | Current work: Player tests and refactoring |

---

## Architecture & Code Locations

- **Game engine:** `engine/game.py`, `engine/player.py`, `engine/constants.py`
- **AlphaBeta bot:** `agents/alphabeta.py`
- **Encoding:** `engine/player.py` (Player._encode), neural net input shape is 168
- **Training:** To be implemented in Phase 1 (supervised trainer)
- **Tests:** `tests/engine/test_player.py` (encoding tests), `tests/engine/test_game.py`

---

## Hard-Won Lessons (Don't Repeat)

From the AlphaZero attempt:
1. **Don't use value-only iterations** — divergence trap
2. **Soft policy targets, not one-hot** — prevents mode collapse
3. **Random agent pretrain is useless** — floor-heavy games
4. **Don't clear buffer after pretrain** — kills value signal
5. **Eval at low sim counts is useless** — need high simulation budgets
6. **Mirror games eliminate factory overfitting** — use them
7. **Round-boundary states must be in training buffer** — net needs to learn them

Supervised approach avoids most of these by construction (no MCTS, no self-play noise).

---

## Critical Gotchas

See [[alphabeta_strategy.md#round_boundaries]] — AlphaBeta must `advance(skip_setup=True)` and read `player.earned` **before** advance mutates the wall.

---

**Status:** Ready to proceed with Phase 0. All docs in place; implementation follows test-driven design.
