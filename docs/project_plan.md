# Azul AlphaZero — Project Plan

> Last updated: 2026-04-03
> Status: Phase 6 — AlphaZero Self-Play Training (in progress) + Phase 6b — Reward Shaping (up next)

---

## Vision

Build a fully playable implementation of the board game **Azul** with an **AlphaZero-style AI opponent**, deployable as a web app and eventually as a mobile app (iOS/Android).

---

## Technology Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.14 | Primary language, best ML ecosystem |
| Front end | FastAPI + HTML/JS | Web-first, iPhone-compatible via Capacitor, shareable by URL |
| Testing | pytest | Industry standard for Python TDD |
| Version control | Git + GitHub | Standard, CI/CD integration |
| CI/CD | GitHub Actions | Free for public repos, integrates natively with GitHub |
| ML framework | PyTorch | Best for custom AlphaZero-style training loops |
| IDE | VS Code | Installed, good Python + git support |

---

## Architecture Overview

```
azul-alphazero/
├── engine/
│   ├── game.py
│   ├── board.py
│   ├── factory.py
│   └── scoring.py
├── agents/
│   ├── base.py
│   ├── random.py
│   ├── cautious.py
│   ├── efficient.py
│   ├── greedy.py
│   ├── mcts.py
│   └── alphazero.py
├── neural/
│   ├── encoder.py
│   ├── model.py
│   ├── trainer.py
│   └── replay.py
├── api/
│   ├── main.py
│   └── schemas.py
├── frontend/
│   ├── index.html
│   ├── game.js
│   └── style.css
├── scripts/
│   ├── self_play.py
│   └── train.py
├── tests/
│   ├── test_game.py
│   ├── test_board.py
│   ├── test_scoring.py
│   ├── test_agents.py
│   ├── test_api.py
│   ├── test_self_play.py
│   ├── test_mcts.py
│   ├── test_encoder.py
│   ├── test_model.py
│   ├── test_replay.py
│   ├── test_trainer.py
│   └── test_alphazero.py
├── checkpoints/     # gitignored
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

### Phase 6 — AlphaZero Self-Play Training 🔄 (in progress)

#### What's built
- `AlphaZeroAgent` — PUCT tree search, value head evaluation, no rollouts
- `collect_self_play` — opponent=None (AZ vs AZ) or opponent=Agent (warmup mode)
- `collect_heuristic_games` — 50% Greedy, 25% Cautious, 25% Efficient, one-hot policy targets
- `scripts/train.py` — full training loop with greedy warmup, auto-switch, per-game eval logging, `_MAX_MOVES=300`

#### Training results so far

| Run | Config | Result | Notes |
|---|---|---|---|
| Test run | 3 iter, 5 games, 20 sim | Gen 0001 | 55% vs prev, 32.5% vs random |
| Overnight 1 | 30 iter, 20 games, 100 sim | Interrupted | Windows sleep |
| Overnight 2 | 30 iter, 20 games, 100 sim, 50k pretrain | Regressed after iter 1 | [0,0] games poisoning buffer |
| Run 4 | 160 iter, 20 games, 100 sim, greedy warmup | 1 generation only | See failure analysis below |

#### Run 4 failure analysis
- Rolling avg score bug: recording 0 for every game where AZ plays as player 1 → warmup threshold never reached → never switched to self-play
- 100 train steps per iteration insufficient — network barely moves each iteration
- 20-game eval too noisy to detect small improvements — new net kept getting reset
- Win vs random stuck at 30-37% throughout — no meaningful learning signal

#### Fixes needed before next run
- Fix rolling average: track win rate vs Greedy rather than raw score, or only count AZ-as-p0 games
- Increase `--train-steps` to 500
- Lower `--win-threshold` to 0.48 for first 20 iterations, or raise `--eval-games` to 40
- Implement reward shaping (see Phase 6b) — deferred scoring is the root cause of slow learning

#### Remaining tasks
- [ ] Fix rolling average bug in `collect_self_play`
- [ ] Increase train steps, tune eval threshold
- [ ] Wire best checkpoint into API `_make_agent()`
- [ ] Add AlphaZero as UI opponent option
- [ ] Elo rating system

---

### Phase 6b — Reward Shaping (up next)

**Motivation:** Azul's scoring is highly deferred. A floor penalty incurred on move 3 isn't applied until end of round (move ~15). A pattern line completed on move 5 doesn't score until end of round. The value head has no way to connect cause and effect across that gap with only 100 simulations. Moving the reward signal closer to the move that earned it should dramatically accelerate learning.

#### Two new engine methods (belong in `engine/scoring.py` or `engine/game.py`)

**`carried_score(board: Board) -> int`**
The official score carried forward from the end of the previous round. This is exactly `board.score` — no calculation needed, just a named accessor for clarity. It is what the scoreboard shows between rounds.

**`earned_score(board: Board, wall_pattern: list[list[Tile]]) -> int`**
Points the player has earned this round but not yet received. Includes:
- Wall placement scores for all currently full pattern lines (calculated as if end-of-round scoring happened now)
- Floor penalties for tiles currently on the floor line
- End-of-game bonuses for any completed rows, columns, or colors already on the wall

The key insight: `earned_score` is deterministic and lossless — once earned, these points cannot be taken away.

**`grand_total(board: Board, wall_pattern) -> int`**
`carried_score + earned_score` — the true picture of a player's position.

#### UI display (builds on engine methods)

Wall tile preview annotations:
- When a pattern line is full, show `+N` on the wall cell where that tile will go
- `N` = wall placement score accounting for all current adjacencies on the wall
- Annotations refresh after every move as adjacencies change

End-of-game bonus indicators:
- Completed column: `+7` below that column
- Completed color: `+10` centered below the row where that color's last tile was placed (between columns)
- Completed row: `+2` to the right of that row

Score display (four values at top of board):
- **Carried** — official score from end of last round (`board.score`)
- **Earned** — points locked in this round not yet applied (`earned_score`)
- **Bonus** — end-of-game bonuses already guaranteed
- **Total** — carried + earned + bonus

#### Model integration
- The model receives only `grand_total` for each player — no breakdown
- `grand_total` is used as the value target in `collect_self_play` instead of final game score
- This gives the value head a signal after every move rather than only at game end
- Implementation: in `collect_self_play`, after each `game.make_move(move)`, compute `grand_total` delta and use as shaped reward blended with final outcome

---

### Phase 7 — Evaluation and Iteration
- [ ] Elo ladder across all agent versions
- [ ] Hyperparameter search
- [ ] Difficulty levels in UI

### Phase 8 — Polish and Release
- [ ] Animated tile placement
- [ ] Sound effects
- [ ] Game history / move replay
- [ ] Cloud deployment
- [ ] Capacitor iOS/Android packaging
- [ ] README with screenshots

---

## Agent Hierarchy

| Agent | Heuristics | Purpose |
|---|---|---|
| `RandomAgent` | None | Benchmark baseline |
| `CautiousAgent` | Floor-avoidance | Avoids penalties |
| `EfficientAgent` | Partial-line preference | Completes lines faster |
| `GreedyAgent` | Both heuristics | Default UI opponent |
| `MCTSAgent` | UCB1 + random rollouts | Lookahead without neural net |
| `AlphaZeroAgent` | PUCT + neural net | Final goal |

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
| 2026-04-03 | Phase 6b defined — carried_score, earned_score, grand_total, UI display |