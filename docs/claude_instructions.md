# Claude Instructions — Azul AlphaZero Project

Paste this into your Claude project instructions so every conversation starts with the right context.

---

## Project context

I am building an Azul board game engine with an AlphaZero AI in Python. This is a learning project — I am a beginner-to-intermediate Python developer. Please explain concepts as you go and don't skip steps.

**Tech stack:**
- Python 3.14, pytest, black, flake8
- FastAPI + HTML/JS frontend
- PyTorch for the neural network
- Git + GitHub, GitHub Actions for CI/CD

**Project structure:** See `PROJECT_PLAN.md` in the repo root for the full folder layout and phase descriptions.

---

## How I want to work

- **TDD always:** Show me the test first. Let me read and understand it. Then show me the implementation.
- **One step at a time:** Don't give me five files at once. Walk me through each piece.
- **Explain the why:** If you make a design decision, tell me why, especially if there's a tradeoff.
- **Git commits:** Remind me when it's a good time to commit, and tell me what the commit message should be.
- **Check CI:** Remind me to push and check that GitHub Actions goes green before moving to the next piece.
- **Repeat on request:** If I ask for something you've already provided, just repeat it without comment.

---

## Current phase

> **Update this whenever a phase is complete.**

Phase 6 — AlphaZero Self-Play Training (up next)

---

## Completed so far

### Phase 1 — Game Engine ✅

Engine modules (`engine/`):
- `tile.py` — Tile enum (5 colors + first-player marker) and COLORS list
- `constants.py` — BOARD_SIZE, PLAYERS, TILES_PER_COLOR, TILES_PER_FACTORY, FLOOR_PENALTIES
- `board.py` — Board dataclass (pattern lines, wall, floor line, score)
- `game_state.py` — GameState dataclass (players, factories, bag, discard, center, round)
- `game.py` — Game class with:
  - `WALL_PATTERN` — fixed 5×5 color grid
  - `wall_column_for(row, color)` — lookup helper
  - `setup_round()` — fills factories, places first-player token in center, increments round
  - `legal_moves()` — returns all valid moves for current player
  - `_is_valid_destination()` — checks wall column specifically, not full row
  - `make_move()` — applies a move, advances current player, triggers score_round + setup_round at end of round
  - `score_round()` — end-of-round scoring, floor penalties, first-player handoff
  - `is_game_over()` — detects completed wall row
  - `score_game()` — end-of-game bonuses (rows, columns, colors)

CLI (`cli/cli.py`):
- Full terminal interface, human vs human
- Displays both boards side by side each turn
- Colored tile symbols with color-blind-friendly distinct letters (B Y R K W)
- Dim colored hints on empty wall cells showing which color belongs there
- Three-prompt move input: color → source → destination
- Accepts numbers or letters for color input

Tests (`tests/`): test_tile.py, test_board.py, test_game_state.py, test_game.py, test_scoring.py

**Known engine gotchas:**
- `_is_valid_destination` checks `player.wall[row][wall_column_for(row, color)] is not None` — not `color in player.wall[row]`. The latter incorrectly blocks colors when any tile is present in the row.
- `_score_floor` must filter out `Tile.FIRST_PLAYER` before adding to discard — first-player markers must never re-enter the bag cycle.
- In a 2-player game, maximum tiles on player boards at any round start is 60 (30 per player: 10 pattern + 20 wall). With 100 tiles in the system, mid-round bag exhaustion is impossible. If `legal_moves()` returns empty mid-round, it is a bug, not an edge case.

---

### Phase 2 — Graphical Front End ✅

Backend (`api/`):
- `schemas.py` — Pydantic models: MoveRequest, BoardResponse, GameStateResponse (includes player_types, round), NewGameRequest, PlayerType
- `main.py` — FastAPI app with endpoints:
  - `GET /state` — return current game state
  - `POST /move` — apply a human move
  - `POST /new-game` — reset with player config (accepts player_types list)
  - `POST /agent-move` — ask current player's agent to move; 422 if it's a human's turn
  - `_make_agent()` — factory function mapping PlayerType strings to agent instances
  - Tie-aware winner logic (winner is None if scores are equal)

Frontend (`frontend/`):
- `index.html` — minimal shell
- `style.css` — full board styling including dialog styles
- `game.js` — full game UI with:
  - `PLAYER_OPTIONS` — defined at module level (not inside a function) so all code can reference it
  - Click-to-select tile interaction (factory → color → destination row)
  - New Game dialog with per-player dropdowns for all agent types
  - Bot turns triggered automatically via `maybeRunBot()` with 600ms delay
  - 2-second inter-round pause when round number changes
  - Tiles disabled during bot turns
  - Player headings use `PLAYER_OPTIONS.find()` lookup for labels
  - "Bot is thinking…" status message during bot turn

---

### Phase 3 — Random Bot + Agent Interface ✅

Agents (`agents/`):
- `base.py` — abstract Agent base class with `choose_move(game) -> Move`
- `random.py` — RandomAgent: true uniform random, no heuristics (standard benchmark)
- `cautious.py` — CautiousAgent: avoids floor if any pattern line move exists
- `efficient.py` — EfficientAgent: prefers placing on partially filled lines
- `greedy.py` — GreedyAgent: both heuristics combined; default UI opponent

Self-play harness (`scripts/`):
- `self_play.py` — CLI script:
  - `run_game(p1, p2) -> GameResult` — plays one complete game
  - `run_series(p1, p2, n) -> SeriesStats` — aggregates N games
  - `AGENT_REGISTRY` — maps string names to agent classes (add new agents here)
  - Progress printed every 5 seconds during long runs
  - Logs results to file via Python logging
  - Usage: `python -m scripts.self_play --games 1000 --p1 greedy --p2 mcts`

Tests (`tests/`): test_agents.py, test_api.py, test_self_play.py

---

### Phase 4 — Monte Carlo Tree Search ✅

Agents (`agents/`):
- `mcts.py` — MCTSAgent with:
  - `ucb1(visits, total_value, parent_visits, c)` — standalone function, keyword-only args
  - `MCTSNode` — holds full Game copy, untried_moves, visits, total_value, parent, children
  - Four MCTS steps: `_select`, `_expand`, `_simulate`, `_backpropagate`
  - `_RandomRolloutAgent` — private class, pure uniform random for rollouts (no heuristics, to avoid biasing value estimates)
  - Simulation result is always from player 0's perspective (1.0 win, 0.0 loss, 0.5 tie)
  - `choose_move` picks most-visited child (not highest UCB1) at decision time

Tests (`tests/`): test_mcts.py — includes `@pytest.mark.slow` strength test (excluded from normal runs via `addopts` in pytest.ini)

**Round-robin benchmark (200 simulations, 100 games per matchup):**

| Rank | Agent | Win rate vs all opponents |
|---|---|---|
| 1 | Greedy | 72.3% |
| 2 | MCTS (200 sim) | 68.5% |
| 3 | Cautious | 58.3% |
| 4 | Efficient | 26.8% |
| 5 | Random | 9.5% |

**Key insight:** GreedyAgent beats MCTSAgent at 200 simulations because Azul's high early-game branching factor (50+ legal moves) means simulations are spread too thin to build reliable value estimates. MCTS becomes significantly stronger in the late game. This is the primary motivation for the neural network policy head in Phase 6 — PUCT focuses simulations on promising moves from the start, bypassing the branching factor problem.

---

### Phase 5 — Neural Network ✅

Neural network modules (`neural/`):
- `encoder.py` — encodes game state as a 116-float normalized vector and moves as integer indices:
  - `encode_state(game) -> Tensor` — always from current player's perspective
  - `encode_move(move, game) -> int` — flat triple index (source × color × destination)
  - `decode_move(index, game) -> Move` — inverse of encode_move
  - `STATE_SIZE = 116`, `MOVE_SPACE_SIZE = 180`
  - Exports offset constants (OFF_MY_WALL, OFF_BAG, etc.) for use in tests and training
- `model.py` — AzulNet: residual MLP with policy and value heads:
  - `ResBlock(dim)` — Linear → LayerNorm → ReLU → Linear → LayerNorm → skip add → ReLU
  - `AzulNet(hidden_dim=256, num_blocks=3)` — stem + trunk + two heads
  - Policy head returns raw logits (softmax applied externally after illegal move masking)
  - Value head returns scalar in (-1, 1) via Tanh
- `replay.py` — ReplayBuffer: circular buffer of (state, policy, value) triples:
  - Pre-allocated tensors, `_pos` cursor wraps with modulo, `_size` capped at capacity
  - `push(state, policy, value)` — overwrites oldest when full
  - `sample(batch_size) -> (states, policies, values)` — random without replacement via `torch.randperm`
- `trainer.py` — Trainer: loss function and training step:
  - `compute_loss(net, states, policies, values)` — MSE value loss + cross-entropy policy loss
  - `Trainer(net, lr=1e-3, batch_size=256)` — Adam optimizer
  - `train_step(buf) -> float` — samples buffer, backpropagates, returns loss
  - `collect_self_play(buf, num_games)` — stubbed, completed in Phase 6

Tests (`tests/`): test_encoder.py, test_model.py, test_replay.py, test_trainer.py

**Key design decisions:**
- Dense float vector over sparse binary planes — trains faster on consumer hardware; Azul's relationships are more combinatorial than spatial
- LayerNorm over BatchNorm — works with any batch size including single-sample inference during MCTS
- Policy head returns logits not probabilities — allows illegal move masking before softmax
- Color index normalized as `(index + 1) / 5` — leaves 0.0 unambiguously meaning "empty pattern line"
- Bag and discard totals included — tile scarcity is strategically important
- First player token encoded as two bits — in center, and whether current player holds it

---

## Conventions in this project

- All Python files use **black** formatting (line length 88)
- Imports sorted by **isort**
- `extend-ignore = E203` in flake8 config — black and flake8 disagree on slice spacing
- Type hints on all function signatures
- Docstrings on all public classes and functions
- Test files live in `tests/` and mirror the source structure
- Test functions named `test_<what it tests>_<condition>`
- American English spelling throughout: "color" not "colour", "center" not "centre"
- Slow tests marked `@pytest.mark.slow` and excluded from default runs via `addopts = -m "not slow"` in `pytest.ini`
- Run slow tests explicitly with: `pytest -m slow -s`

---

## Things Claude should never do in this project

- Don't write engine code that imports from `api/` or `frontend/`
- Don't skip writing tests — if there's no test, the code doesn't exist yet
- Don't use `print()` for debugging in engine code — use Python's `logging` module
- Don't give me a huge block of code to copy-paste without explaining it first