# Comprehensive Training & Model Design
## Pretrain Mode + Temperature Exploration + Mirror Eval + Policy Split + Value Simplification

## Overview

This document outlines five integrated changes to the AlphaZero training pipeline and model architecture:

### Training Changes
1. **Pretrain mode** (`--pretrain` flag): Flood buffer with weak mirrored heuristic games, train until loss plateaus, then auto-transition to normal AlphaZero self-play
2. **Temperature exploration** (25-move rule): First 25 moves of each game use high temperature (exploration), remaining moves use low temperature (deterministic)
3. **Mirror eval**: Evaluate by playing both sides of each matchup (with same seed) to reduce factory-seed variance

### Model Architecture Changes
4. **Three-head policy** (instead of flat 210-value head):
   - Source head (2 values): center vs factory
   - Tile head (5 values): which color
   - Destination head (6 values): where to place
   - Move probability = source_prob × tile_prob × destination_prob

5. **Value head simplification**: Remove value_abs from training loss (keep in model as diagnostic only)

---

## Part 1: Training Features (Pretrain + Temperature + Mirror Eval)

### 1.1 Pretrain Mode

**Purpose:** Quick value head calibration to avoid floor moves and 100-move games

**Mechanism:**
- Flag: `train.py --pretrain`
- Generates weak mirrored heuristic games (random/efficient/cautious/greedy vs easy)
- Trains until loss plateaus (first time loss increases after initial iterations)
- Automatically switches to normal AlphaZero self-play
- Buffer carries over to phase 2

**Hardcoded constants (not runtime-editable):**
```python
PRETRAIN_GAMES_PER_ITER = 100      # Mirror pairs per iteration
PRETRAIN_TRAINING_STEPS = 500       # Gradient steps per iteration
PRETRAIN_MAX_ITERATIONS = 50        # Safety limit before force-stop
PRETRAIN_PLATEAU_SKIP_EARLY = 2    # Skip first N iterations for plateau detection
```

### 1.2 Temperature Exploration (25-Move Rule)

**Rule:** 
- Moves 0-24 (by `game.turn` global counter): high temperature (explorative)
- Moves 25+: low temperature (deterministic)

**Why:** Prevents deterministic cycles in early game, keeps endgames sharp.

**Hardcoded constants:**
```python
EXPLORATION_MOVES = 25
EXPLORATION_TEMP = 1.0
DETERMINISTIC_TEMP = 0.1
```

### 1.3 Mirror Eval

**Rule:** For each evaluation, play both sides of the matchup with identical seed.

- Game 1: new_net p0, best_net p1, seed=S
- Game 2: best_net p0, new_net p1, seed=S (same seed)
- Count wins from both games

**Skip mirroring when:** Both agents are instances of deterministic algorithms (e.g., AlphaBeta vs AlphaBeta)

**Why:** Factory seed can correlate with outcome. Playing both sides neutralizes this bias.

---

## Part 2: Model Architecture (Three-Head Policy + Value Simplification)

### 2.1 Three-Head Policy Architecture

**Current:** Single policy head outputs 210 values (one per move = source × tile × destination)
```
MOVE_SPACE_SIZE = NUM_SOURCES (7) × BOARD_SIZE (5) × NUM_DESTINATIONS (6) = 210
```

**New:** Three separate heads that combine multiplicatively
```python
source_head:      2 values (center, factory)
tile_head:        5 values (colors)
destination_head: 6 values (destinations)

move_prob[source][tile][dest] = source_prob[source] × tile_prob[tile] × dest_prob[dest]
```

**Why separate heads:**
- Network learns what matters: tile selection and destination placement
- Source is mostly "do I want center?" (for first player token)
- Reduces overfitting by factorizing the policy space
- Simpler loss targets from MCTS visits

### 2.2 Move Resolution Logic

**Given a sampled (source_type, tile, destination):**

```python
# 1. Determine which actual sources have this tile
sources_with_tile = [s for s in all_sources if tile in s]

# 2. If source_type == CENTER and center has tile
if source_type == CENTER and CENTER in sources_with_tile:
    return Move(source=CENTER, tile=tile, destination=destination)

# 3. Else: pick from available factories (in order or random)
factory_sources = [s for s in sources_with_tile if s != CENTER]
if factory_sources:
    source = factory_sources[0]  # or random.choice(factory_sources)
    return Move(source=source, tile=tile, destination=destination)

# 4. No valid move (shouldn't happen in legal move generation)
```

**MCTS behavior:** Explores all valid (source_type, tile, destination) combinations, resolving each to a full move via above logic.

### 2.3 Value Head Simplification

**Current:**
- Three value heads: value_win, value_diff, value_abs
- All three included in training loss
- Loss = policy_loss + 0.3×value_win_loss + 1.0×value_diff_loss + 0.1×value_abs_loss

**New:**
- Keep all three heads in model (for diagnostics)
- Remove value_abs from training loss
- Loss = policy_loss + 0.3×value_win_loss + 1.0×value_diff_loss
- value_abs is still computed and logged, but not used for training

**Why:** value_abs (absolute score prediction) is mostly noise; the other two heads (win/diff) are what matter for training.

---

## Part 3: Implementation Plan

### Files to Modify

1. **neural/model.py** (AzulNet architecture)
   - Split policy_head into source_head, tile_head, destination_head
   - Keep value_abs_head but exclude from loss computation

2. **neural/encoder.py** (move encoding/decoding)
   - Add functions to convert between new 3-head format and flat MOVE_SPACE_SIZE format
   - Update policy target generation to work with split heads

3. **neural/trainer.py** (training & game generation)
   - Add `_pretrain_matchups()` function
   - Add move_count parameter to game generation
   - Modify loss computation to exclude value_abs
   - Update policy target handling for 3-head format

4. **neural/search_tree.py** (MCTS)
   - Add move_count parameter to `_pick_move()`
   - Implement temperature rule (25-move cutoff)
   - Update policy sampling to use 3-head format
   - Implement move resolution (source_type + tile + dest → full move)

5. **agents/alphazero.py** (self-play agent)
   - Thread move_count through game loop
   - Pass move_count to SearchTree

6. **scripts/train.py** (main training loop)
   - Add `--pretrain` flag
   - Add pretrain constants
   - Implement plateau detection
   - Branch on pretrain vs normal mode
   - Modify eval to use mirror games
   - Remove value_abs from loss computation

---

## Implementation Steps (Ordered by Dependency)

### Step 1: Update AzulNet Architecture (neural/model.py)

**Change policy head:**
```python
# OLD
self.policy_head = nn.Linear(hidden_dim, MOVE_SPACE_SIZE)

# NEW
self.source_head = nn.Linear(hidden_dim, 2)         # center vs factory
self.tile_head = nn.Linear(hidden_dim, 5)           # colors
self.destination_head = nn.Linear(hidden_dim, 6)    # destinations
```

**Return signature update:**
```python
def forward(self, encoding: torch.Tensor) -> tuple[...]:
    trunk = self.blocks(self.input_proj(encoding))
    
    return (
        (self.source_head(trunk), self.tile_head(trunk), self.destination_head(trunk)),
        self.value_win_head(trunk),
        self.value_diff_head(trunk),
        self.value_abs_head(trunk),  # kept for diagnostics
    )

# Returns: ((source_logits, tile_logits, dest_logits), value_win, value_diff, value_abs)
```

### Step 2: Add Helper Functions to encoder.py

**Convert between 3-head and flat formats:**
```python
def policy_3head_to_flat(source_probs, tile_probs, dest_probs) -> torch.Tensor:
    """Convert (2, 5, 6) head outputs to (210,) flat policy."""
    # Multiply probabilities: flat[s][t][d] = source_probs[s] * tile_probs[t] * dest_probs[d]
    pass

def flat_to_policy_3head(flat_policy) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert (210,) flat policy back to 3-head format (approximately)."""
    # For MCTS visit-count aggregation: marginalize to per-head probabilities
    pass
```

### Step 3: Add `_pretrain_matchups()` to trainer.py

```python
def _pretrain_matchups() -> list[MatchupSpec]:
    """Weak heuristics vs easy for fast buffer initialization."""
    def make_random() -> RandomAgent:
        return RandomAgent()
    def make_efficient() -> EfficientAgent:
        return EfficientAgent()
    def make_cautious() -> CautiousAgent:
        return CautiousAgent()
    def make_greedy() -> GreedyAgent:
        return GreedyAgent()
    def make_easy() -> AlphaBetaAgent:
        return AlphaBetaAgent(depths=(1, 2, 3), thresholds=(20, 10))
    
    return [
        (make_random, make_easy, 0.25),
        (make_efficient, make_easy, 0.25),
        (make_cautious, make_easy, 0.25),
        (make_greedy, make_easy, 0.25),
    ]
```

### Step 4: Update Loss Computation in trainer.py

**Modify `compute_loss()`:**
```python
def compute_loss(...) -> dict[str, float]:
    # ... existing code ...
    
    # Policy loss (unchanged)
    policy_loss = ...
    
    # Value losses
    value_win_loss = F.mse_loss(value_win_pred, value_win_target)
    value_diff_loss = F.mse_loss(value_diff_pred, value_diff_target)
    # NOTE: value_abs_loss is NOT included
    
    # Combined loss
    value_loss = 0.3 * value_win_loss + 1.0 * value_diff_loss  # no value_abs
    total_loss = policy_loss + value_loss
    
    return {
        "total": total_loss,
        "policy": policy_loss,
        "value": value_loss,
        "value_win": value_win_loss,
        "value_diff": value_diff_loss,
        "value_abs": value_abs_loss,  # still computed for logging, but not in total
    }
```

### Step 5: Update SearchTree for 3-Head Policy & Temperature

**Modify `neural/search_tree.py`:**

Add constants:
```python
EXPLORATION_MOVES = 25
EXPLORATION_TEMP = 1.0
DETERMINISTIC_TEMP = 0.1

NUM_SOURCES = 2       # center vs factory
NUM_TILES = 5
NUM_DESTINATIONS = 6
```

Update policy application in `add_root_exploration()`:
```python
def add_root_exploration(self, policy: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Add Dirichlet noise to 3-head policy at root."""
    source_logits, tile_logits, dest_logits = policy
    # Apply softmax & Dirichlet noise to each head
    source_probs = softmax(source_logits) + dirichlet_noise(...)
    tile_probs = softmax(tile_logits) + dirichlet_noise(...)
    dest_probs = softmax(dest_logits) + dirichlet_noise(...)
    self.root_policy = (source_probs, tile_probs, dest_probs)
```

Update `_pick_move()` to accept move_count and implement temperature rule:
```python
def _pick_move(self, move_count: int) -> Move:
    """Pick move based on visit counts, with temperature rule."""
    # Calculate temperature: high for first 25 moves, low after
    temperature = EXPLORATION_TEMP if move_count < EXPLORATION_MOVES else DETERMINISTIC_TEMP
    
    # Sample from each head's visit distribution
    source_visits = [child.visit_count for child in self.children if child.source_type == s]
    tile_visits = [child.visit_count for child in self.children if child.tile == t]
    dest_visits = [child.visit_count for child in self.children if child.destination == d]
    
    # Apply softmax with temperature to visits
    source_type = sample_softmax(source_visits, temperature)
    tile = sample_softmax(tile_visits, temperature)
    destination = sample_softmax(dest_visits, temperature)
    
    # Resolve to full move
    return resolve_move(source_type, tile, destination)
```

Add move resolution:
```python
def resolve_move(source_type: int, tile: Tile, destination: Destination) -> Move:
    """Resolve (source_type, tile, destination) to full Move."""
    sources_with_tile = [s for s in self.game.available_sources() if tile in s]
    
    if source_type == CENTER and CENTER in sources_with_tile:
        return Move(source=CENTER, tile=tile, destination=destination)
    
    factory_sources = [s for s in sources_with_tile if s != CENTER]
    if factory_sources:
        source = factory_sources[0]  # in-order selection
        return Move(source=source, tile=tile, destination=destination)
    
    raise ValueError(f"No valid source for tile {tile}")
```

### Step 6: Add Move Count Tracking to AlphaZeroAgent

**Modify `agents/alphazero.py`:**
```python
def _play_game(self, ...) -> list[...]:
    move_count = 0
    while not game.is_over() and move_count < 100:
        # ... get current player ...
        if current_player is self:
            move = self.pick_move(game, move_count)  # pass move_count
        else:
            move = opponent.pick_move(game)
        
        game.make_move(move)
        move_count += 1
        # ... record history ...
```

Update SearchTree call:
```python
def pick_move(self, game: Game, move_count: int = 0) -> Move:
    tree = SearchTree(game, ...)
    for _ in range(self.simulations):
        tree.search(...)
    return tree._pick_move(move_count)  # pass move_count
```

### Step 7: Add Pretrain Constants and Flag to train.py

```python
# Pretrain mode parameters (hardcoded, not runtime-editable)
PRETRAIN_GAMES_PER_ITER = 100
PRETRAIN_TRAINING_STEPS = 500
PRETRAIN_MAX_ITERATIONS = 50
PRETRAIN_PLATEAU_SKIP_EARLY = 2

# Temperature exploration
EXPLORATION_MOVES = 25
EXPLORATION_TEMP = 1.0
DETERMINISTIC_TEMP = 0.1
```

Add argparse flag:
```python
parser.add_argument("--pretrain", action="store_true",
                    help="Run pretrain mode: fill buffer with weak heuristics, train until loss plateaus")
```

### Step 8: Implement Plateau Detection in train.py

```python
prev_loss = None
phase2_started = False

for iteration in range(args.iterations):
    # ... training code ...
    current_loss = accumulated_loss / num_steps
    
    if args.pretrain and not phase2_started:
        # Check for plateau
        if prev_loss is not None and iteration > PRETRAIN_PLATEAU_SKIP_EARLY:
            if current_loss > prev_loss:
                logger.info("Pretrain loss plateau detected. Switching to self-play.")
                save_checkpoint(net, generation=0, path="checkpoints/pretrain_final.pt")
                phase2_started = True
    
    prev_loss = current_loss
```

### Step 9: Modify Main Training Loop for Pretrain vs Normal Mode

```python
phase2_started = False

for iteration in range(args.iterations):
    if args.pretrain and not phase2_started:
        # PRETRAIN PHASE: weak heuristics, mirror games
        logger.info("Pretrain iteration %d/%d", iteration + 1, args.iterations)
        
        collect_mirror_heuristic_games(
            buf,
            num_pairs=PRETRAIN_GAMES_PER_ITER,
            matchups=_pretrain_matchups(),
        )
        
        # Train (same as normal training)
        # ... training steps ...
        
        # Check plateau
        if iteration > PRETRAIN_PLATEAU_SKIP_EARLY and current_loss > prev_loss:
            phase2_started = True
    
    elif args.pretrain and phase2_started:
        # PHASE 2: Normal self-play (same as no-pretrain)
        logger.info("Phase 2: AlphaZero self-play iteration %d", iteration + 1)
        collect_self_play(buf, net, ...)
        # ... training and eval ...
    
    else:
        # NO PRETRAIN: normal AlphaZero from start
        collect_self_play(buf, net, ...)
        # ... training and eval ...
```

### Step 10: Update Eval for Mirror Games

```python
def evaluate(new_net, best_net, num_games=20, ...):
    """Evaluate with optional mirroring."""
    should_mirror = not are_deterministic_bots(new_net, best_net)
    
    new_wins = 0
    total_games = 0
    
    for game_i in range(num_games):
        seed = random.randint(0, 2**31)
        
        # Game 1: new p0, best p1
        game = Game(seed=seed)
        game.setup_round()
        # ... play game ...
        new_wins += (scores[0] > scores[1])
        total_games += 1
        
        # Game 2: mirrored (if applicable)
        if should_mirror:
            game = Game(seed=seed)  # same seed!
            game.setup_round()
            # ... play with sides swapped ...
            new_wins += (scores[1] > scores[0])  # new is p1 now
            total_games += 1
    
    return new_wins / total_games

def are_deterministic_bots(agent1, agent2) -> bool:
    from agents.alphabeta import AlphaBetaAgent
    return isinstance(agent1, AlphaBetaAgent) and isinstance(agent2, AlphaBetaAgent)
```

---

## Testing Strategy

### 1. Model Architecture
```bash
# Test that new model loads and runs
python -c "from neural.model import AzulNet; net = AzulNet(); print('Model OK')"

# Test policy output format
import torch
net = AzulNet()
encoding = torch.randn(1, 123)
(source, tile, dest), value_win, value_diff, value_abs = net(encoding)
assert source.shape == (1, 2)
assert tile.shape == (1, 5)
assert dest.shape == (1, 6)
```

### 2. Pretrain Mode
```bash
python -m scripts.train --pretrain --iterations 20
# Expected logs:
#   - Pretrain mode activated
#   - generating 100 mirror game pairs (multiple times)
#   - Loss decreasing
#   - "Pretrain loss plateau detected..." around iteration 2-5
#   - "Switching to self-play mode"
#   - Normal AlphaZero continues
```

### 3. Temperature Rule
Manual verification via policy sampling:
- Moves 0-24: high entropy (more uniform)
- Moves 25+: low entropy (peaked)

### 4. Mirror Eval
Check logs show both sides:
- "eval game 1/20: ...new+, new_win_rate=100%"
- "eval game 2/20: ...old-, new_win_rate=50%"
- (Both games played per matchup)

### 5. Value Loss Simplification
Check training loss logs:
- Should NOT include value_abs in total loss
- value_abs should still be logged as diagnostic
- Loss formula should show: policy + 0.3×value_win + 1.0×value_diff

---

## Checkpoint Compatibility

**Breaking change:** New model (3-head policy) is NOT compatible with old checkpoints (flat policy).

**Migration:**
- Can't load old checkpoints into new model directly
- Would need to initialize new model from scratch or implement conversion logic
- Alternatively: train the new model from scratch (pretrain will catch it up)

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `neural/model.py` | Split policy head (3 heads), keep value_abs |
| `neural/encoder.py` | Add 3-head ↔ flat conversion helpers |
| `neural/trainer.py` | Add `_pretrain_matchups()`, update loss, thread move_count |
| `neural/search_tree.py` | Add temperature rule, move resolution, 3-head policy sampling |
| `agents/alphazero.py` | Thread move_count through game loop |
| `scripts/train.py` | Add `--pretrain`, plateau detection, main loop branching, mirror eval |

---

## Success Criteria

- ✓ New AzulNet loads and produces 3-head policy output
- ✓ `train.py --pretrain` fills buffer with weak heuristics
- ✓ Loss plateaus within first 5 iterations
- ✓ Auto-transition to self-play mode
- ✓ Buffer carries over (no clearing)
- ✓ Temperature rule applied (25 moves high, rest low)
- ✓ Mirror eval plays both sides with same seed
- ✓ value_abs excluded from training loss but still logged
- ✓ No regression: `train.py` without `--pretrain` works as before
- ✓ New checkpoint saved at pretrain→phase2 transition

---

## Known Implementation Details

1. **Move count tracking:** Pass `game.turn` through AlphaZeroAgent → SearchTree pipeline

2. **Source resolution:** When (source_type, tile, dest) is sampled, look up actual sources with that tile, prioritize center

3. **Policy target generation:** MCTS visits aggregated to 3 heads (marginalizing over unused dimensions)

4. **Seed propagation:** Verify `Game(seed=S)` works for self-play eval

5. **Loss computation:** value_abs still computed for logging, but coefficient is 0 in total loss
