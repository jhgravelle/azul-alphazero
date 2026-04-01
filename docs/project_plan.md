# Azul AlphaZero — Project Plan

> Last updated: 2026-04-01
> Status: Phase 4 — Monte Carlo Tree Search (up next)

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
│   ├── factory.py   # Factory displays and centre pool
│   └── scoring.py   # End-of-round and end-of-game scoring
├── agents/          # AI agent implementations
│   ├── base.py      # Abstract Agent interface
│   ├── random.py    # Random move agent (with heuristics)
│   ├── mcts.py      # Pure MCTS agent
│   └── alphazero.py # Neural net + MCTS agent
├── neural/          # PyTorch model and training
│   ├── model.py     # ResNet policy + value network
│   ├── trainer.py   # Self-play and training loop
│   └── replay.py    # Experience replay buffer
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
│   └── ...
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
- [x] Implement `RandomAgent` with basic heuristics (avoids floor, prefers partial lines)
- [x] Wire agent into the game loop via `/agent-move` API endpoint
- [x] New Game dialog with per-player dropdowns (Human / Random Bot)
- [x] Bot turns trigger automatically in the UI with inter-round pause
- [x] Add a self-play harness (`scripts/self_play.py`) — bot vs bot, N games
- [x] Log game statistics (win rate, game length, score distributions)
- [x] Progress indicator printed every 5 seconds during long runs
- [x] `AGENT_REGISTRY` for easy addition of future agents

---

### Phase 4 — Monte Carlo Tree Search
*Goal: a competent bot using pure MCTS (no neural net yet)*

- [ ] Implement MCTS with UCB1 selection
- [ ] Implement random rollout policy
- [ ] Tune simulation count vs. play speed
- [ ] `MCTSAgent` beats `RandomAgent` >80% of the time
- [ ] Add MCTS bot as an opponent option in the UI
- [ ] Add MCTS to `AGENT_REGISTRY` in self-play harness

**Definition of done:** MCTSAgent wins >80% vs RandomAgent over 200 games. A human can play against it in the browser.

---

### Phase 5 — Neural Network
*Goal: a trained policy + value network for Azul*

- [ ] Design Azul state encoding (feature vector / tensor)
- [ ] Build ResNet architecture with policy head and value head
- [ ] Write training loop with supervised warm-up on MCTS data
- [ ] Verify network learns (loss decreases, value head calibration)
- [ ] Unit tests for state encoding (no silent shape bugs)

**Definition of done:** Network trains without errors, loss curves look healthy, value predictions correlate with actual outcomes.

---

### Phase 6 — AlphaZero Self-Play Training
*Goal: iterative self-play that produces a strong Azul AI*

- [ ] Replace MCTS rollouts with neural net value estimates
- [ ] Self-play data generation loop
- [ ] Experience replay buffer
- [ ] Iterative training: generate data → train → evaluate → keep if better
- [ ] Model checkpointing
- [ ] Elo rating system to track model strength over generations

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
| 2026-04-01 | Phases 1, 2, 3 complete. Phase 4 up next. |