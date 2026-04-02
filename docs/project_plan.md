# Azul AlphaZero — Project Plan

> Last updated: 2026-04-02
> Status: Phase 6 — AlphaZero Self-Play Training (up next)

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
├── engine/          # Pure Python game logic (no UI dependencies)
│   ├── game.py      # Game state, rules, legal moves
│   ├── board.py     # Player board, pattern lines, wall
│   ├── factory.py   # Factory displays and center pool
│   └── scoring.py   # End-of-round and end-of-game scoring
├── agents/          # AI agent implementations
│   ├── base.py      # Abstract Agent interface
│   ├── random.py    # True uniform random agent (baseline)
│   ├── cautious.py  # Floor-avoidance heuristic only
│   ├── efficient.py # Partial-line preference heuristic only
│   ├── greedy.py    # Both heuristics (default UI opponent)
│   ├── mcts.py      # Pure MCTS agent (UCB1)
│   └── alphazero.py # Neural net + MCTS agent (Phase 6)
├── neural/          # PyTorch model and training
│   ├── encoder.py   # State vector and move index encoding
│   ├── model.py     # Residual MLP policy + value network
│   ├── trainer.py   # Loss function and training step
│   └── replay.py    # Circular experience replay buffer
├── api/             # FastAPI web server
│   ├── main.py      # App entry point, routes
│   └── schemas.py   # Pydantic request/response models
├── frontend/        # HTML/JS/CSS game UI
│   ├── index.html
│   ├── game.js
│   └── style.css
├── scripts/         # Standalone utilities
│   └── self_play.py # Self-play harness CLI
├── tests/           # pytest test suite
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
│   └── test_trainer.py
├── docs/            # Project documentation
│   └── PROJECT_PLAN.md  (this file)
├── .github/
│   └── workflows/
│       └── ci.yml   # GitHub Actions CI pipeline
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## Development Phases

### Phase 0 — Project Setup ✅ (complete)
*Goal: professional project skeleton before any game code*

- [x] Create GitHub repository
- [x] Clone to local machine
- [x] Create and activate Python virtual environment
- [x] Install dev dependencies (pytest, black, flake8, isort, pytest-watch)
- [x] Create folder structure
- [x] Configure pytest (`pytest.ini`)
- [x] Configure code formatter (`pyproject.toml`)
- [x] Set up GitHub Actions CI (runs tests on every push)
- [x] Write and pass first dummy test (proves CI works)
- [x] Commit and push — CI goes green

---

### Phase 1 — Game Engine ✅ (complete)
*Goal: a complete, fully-tested Azul rule engine with no UI*

- [x] Model game state as Python dataclasses
- [x] Implement factory display setup and tile drawing
- [x] Implement legal move generation
- [x] Implement tile placement (pattern lines → wall)
- [x] Implement end-of-round scoring
- [x] Implement end-of-game bonus scoring
- [x] Implement game-over detection
- [x] Text-based CLI so a human can play both sides
- [x] Full test suite — every rule covered

---

### Phase 2 — Graphical Front End ✅ (complete)
*Goal: a proper visual game board in the browser*

- [x] FastAPI server that serves the frontend and exposes a game API
- [x] HTML/JS frontend that renders the full Azul board
- [x] Clicking tiles and factory displays makes legal moves
- [x] Game state updates are reflected visually
- [x] Human vs human (passing the keyboard)
- [x] Clean separation: UI calls API, API calls engine

---

### Phase 3 — Random Bot + Agent Interface ✅ (complete)
*Goal: a pluggable agent system and a baseline opponent*

- [x] Define abstract `Agent` base class with a `choose_move(game_state)` method
- [x] Implement `RandomAgent` (true uniform random — standard benchmark baseline)
- [x] Wire agent into the game loop via `/agent-move` API endpoint
- [x] New Game dialog with per-player dropdowns
- [x] Bot turns trigger automatically in the UI with inter-round pause
- [x] Add a self-play harness (`scripts/self_play.py`) — bot vs bot, N games
- [x] Log game statistics (win rate, game length, score distributions)
- [x] `AGENT_REGISTRY` for easy addition of future agents

---

### Phase 4 — Monte Carlo Tree Search ✅ (complete)
*Goal: a competent bot using pure MCTS (no neural net yet)*

- [x] Implement MCTS with UCB1 selection
- [x] Implement random rollout policy
- [x] `MCTSAgent` beats true `RandomAgent` >80% over 200 games (achieved 94%)
- [x] Add MCTS bot as an opponent option in the UI
- [x] Refactor heuristic agents into distinct classes: `CautiousAgent`, `EfficientAgent`, `GreedyAgent`
- [x] All agents available in New Game dialog and `AGENT_REGISTRY`
- [x] Round-robin benchmark: 100 games per matchup, 200 simulations

**Round-robin results (200 simulations, 100 games each):**

| Rank | Agent | Win rate vs all opponents |
|---|---|---|
| 1 | Greedy | 72.3% |
| 2 | MCTS (200 sim) | 68.5% |
| 3 | Cautious | 58.3% |
| 4 | Efficient | 26.8% |
| 5 | Random | 9.5% |

**Known limitation of pure MCTS:** Azul has a high branching factor in early rounds (50+ legal moves). At 200 simulations, each child of the root receives only a few visits, so value estimates are too noisy to overcome a well-tuned heuristic opponent. MCTS earns its strength in the late game when branching factor drops and rollouts become more informative. This is the primary motivation for the neural network policy head in Phase 6 — it concentrates simulations on promising moves immediately, bypassing the branching factor problem.

---

### Phase 5 — Neural Network ✅ (complete)
*Goal: a trained policy + value network for Azul*

- [x] Design Azul state encoding — dense 116-float normalized vector
- [x] Build residual MLP with policy head and value head
- [x] Implement circular experience replay buffer
- [x] Implement loss function (MSE value loss + cross-entropy policy loss)
- [x] Implement training step with Adam optimizer
- [x] Unit tests for all neural network components

**State vector layout (116 floats, all normalized to [0, 1]):**

| Section | Size | Encoding |
|---|---|---|
| My wall | 25 | 1.0/0.0 per cell, row-major |
| Opponent wall | 25 | same |
| My pattern line fill ratios | 5 | tiles present / row capacity |
| My pattern line colors | 5 | color index / 5, or 0.0 if empty |
| Opponent pattern line fill ratios | 5 | same |
| Opponent pattern line colors | 5 | same |
| Factories | 25 | count of each color per factory / 4 |
| Center color counts | 5 | count / TILES_PER_COLOR |
| First player token in center | 1 | 1.0 / 0.0 |
| I hold first player token | 1 | 1.0 / 0.0 |
| My floor | 1 | tiles on floor / 7 |
| Opponent floor | 1 | same |
| My score | 1 | score / 100 |
| Opponent score | 1 | same |
| Bag totals | 5 | count of each color / 20 |
| Discard totals | 5 | count of each color / 20 |

Always encoded from the current player's perspective — "my" = current player, "opp" = other player.

**Network architecture:**
- Input: 116-float vector
- Stem: Linear(116 → 256) → LayerNorm → ReLU
- Trunk: 3 × ResBlock(256) — each block: Linear → LayerNorm → ReLU → Linear → LayerNorm → skip add → ReLU
- Policy head: Linear(256 → 180) — raw logits; softmax applied externally after masking illegal moves
- Value head: Linear(256 → 64) → ReLU → Linear(64 → 1) → Tanh — scalar in (-1, 1)

**Move space:** 180 indices = 6 sources × 5 colors × 6 destinations (5 pattern lines + floor).

**Key design decisions:**
- Dense float vector over sparse binary planes — trains faster on consumer hardware; Azul's relationships are more combinatorial than spatial
- LayerNorm over BatchNorm — BatchNorm requires batch size > 1, LayerNorm works with any batch size including single-sample inference
- Logits not probabilities from policy head — allows illegal move masking before softmax
- `collect_self_play()` stubbed in `trainer.py` — will be completed in Phase 6 once `AlphaZeroAgent` exists

---

### Phase 6 — AlphaZero Self-Play Training
*Goal: iterative self-play that produces a strong Azul AI*

- [ ] Implement `AlphaZeroAgent` — PUCT tree search guided by the neural network
- [ ] Replace UCB1 with PUCT (adds prior probability from policy head to exploration term)
- [ ] Replace random rollouts with neural net value estimates
- [ ] Complete `collect_self_play()` in `trainer.py`
- [ ] Iterative training loop: generate data → train → evaluate → keep if better
- [ ] Model checkpointing (save/load network weights)
- [ ] Elo rating system to track model strength over generations
- [ ] Add AlphaZero agent as a UI opponent option

**Definition of done:** Generation N+1 beats Generation N at a statistically significant rate (>55% over 200 games).

---

### Phase 7 — Evaluation and Iteration
*Goal: a measurably strong, tunable AI*

- [ ] Elo ladder across all agent versions
- [ ] Hyperparameter search (MCTS simulations, network size, learning rate)
- [ ] Add difficulty levels in the UI (maps to MCTS simulation count)
- [ ] Optional: opening book or curriculum learning

---

### Phase 8 — Polish and Release
*Goal: something you'd be proud to share*

- [ ] Animated tile placement
- [ ] Sound effects (optional)
- [ ] Game history / move replay
- [ ] Deploy to a cloud host (Render, Railway, or Fly.io — all have free tiers)
- [ ] Capacitor packaging for iOS / Android App Store
- [ ] README with screenshots and instructions
- [ ] Show round number in the UI header
- [ ] Label winner more clearly on game over (highlight winning board)
- [ ] Bot vs bot: inter-round pause timing polish

---

## Agent Hierarchy

| Agent | Heuristics | Purpose |
|---|---|---|
| `RandomAgent` | None — true uniform random | Standard benchmark baseline |
| `CautiousAgent` | Floor-avoidance only | Avoids penalties, no planning |
| `EfficientAgent` | Partial-line preference only | Completes lines faster, ignores floor cost |
| `GreedyAgent` | Both heuristics | Strongest heuristic agent; default UI opponent |
| `MCTSAgent` | UCB1 tree search + random rollouts | Lookahead without a neural net |
| `AlphaZeroAgent` | PUCT tree search + neural net | Final goal |

`GreedyAgent` is the recommended default opponent for human players. It is stronger than `MCTSAgent` at low simulation counts due to Azul's high early-game branching factor.

---

## Key Principles

**Test-Driven Development (TDD):** Write the failing test first, then write the code to make it pass. Never write engine code without a test.

**Engine independence:** The game engine (`engine/`) must never import from `api/` or `frontend/`. It is pure Python logic. This is what makes it testable, and what will make the AI training fast.

**Commit often:** A commit should represent one coherent thing ("add end-of-round scoring", "fix tile draw bug"). If your commit message needs the word "and" more than once, split it up.

**CI is the source of truth:** If GitHub Actions is red, the branch is broken. Don't move forward until it's green.

---

## Reference Links

- [Azul rulebook (PDF)](https://www.nextmovegames.com/en/index.php?controller=attachment&id_attachment=11)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [FastAPI docs](https://fastapi.tiangolo.com/)
- [PyTorch docs](https://pytorch.org/docs/)
- [GitHub Actions docs](https://docs.github.com/en/actions)

---

## Change Log

| Date | Change |
|---|---|
| 2026-03-29 | Initial project plan created |
| 2026-04-01 | Phases 1–3 complete |
| 2026-04-01 | Phase 4 complete — MCTSAgent, heuristic agent refactor, round-robin benchmark |
| 2026-04-02 | Phase 5 complete — encoder, AzulNet, replay buffer, trainer |