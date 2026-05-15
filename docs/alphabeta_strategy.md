# AlphaBeta Strategy — Why It Works as Training Target

---

## Overview

**AlphaBeta agent with earned-score heuristic is the best performing baseline bot.** It beats Minimax decisively (76% win rate) and plateaued AlphaZero self-play (55%+ win rate). We use AlphaBeta as our training oracle: its game outcomes label our supervised learning dataset.

---

## Why AlphaBeta Wins

### Heuristic Strength: Earned Score
AlphaBeta evaluates leaf nodes using:
```
position_value = (my_earned_score - opponent_earned_score) / 50.0
```

**Why this works:**
- Earned score = score + pending + penalty + bonus
- Incentivizes completing tiles (pending) and avoiding floor (penalty)
- Bonuses encourage diagonal/row/col combos
- Single-feature simplicity → no overfitting, generalizes across positions

### Depth Advantage
AlphaBeta searches 2–4 moves ahead (source-adaptive):
- 1–2 moves in early round (50+ legal moves)
- 3–4 moves in late round (5–10 legal moves)

**Effect:** Sees round-end scoring outcomes (0–2 moves), which is where earned-score heuristic is most accurate.

### Determinism
Stochastic move selection (softmax temperature 0.3) prevents deterministic loops but remains reproducible with seed control. Identical games (same seed) produce identical scores despite randomness.

---

## Design Constraint: Round Boundaries

### What Round Boundaries Mean

**Round boundary = position at end of a round, after all players have moved but before score_round() executes.**

When AlphaBeta searches, leaves are **always at round boundaries**:
- Factories empty
- Pattern lines committed
- Scores pending (not yet applied)

AlphaBeta does NOT search across `advance()` — it calls `advance(skip_setup=True)` only when backtracking the value up the tree, never to generate children.

### Why This Constraint?

#### 1. Prevents Expensive Round-End Operations Inside Tree
`score_round()` computes:
- Adjacency bonuses (wall traversal per cell)
- Floor penalties
- Wall placement scoring
- These are 10–20ms each in a 50-move tree search

Keeping leaves at round boundaries = small trees, fast search.

#### 2. Forces Value Net to Generalize
End-of-round state encodes:
- Final pattern lines for the round
- Pending scores (not yet applied)
- Opponent's state (unchanged)

Next round setup:
- Factories refilled (random from bag)
- First player token moved
- Scoring applied

**If value net memorized factory setups,** it would fail when next round has different tiles. By staying at round boundaries, we force the net to evaluate "will this wall arrangement and score combo lead to victory?" without relying on lucky next-round factories.

#### 3. Keeps Bot Context Tractable
With round boundaries:
- Maximum tree depth ≈ 5 moves per round × 6 rounds = 30 total moves
- Actual depths: 1–4 moves (Azul games don't reach all 6 rounds)
- Tree size: 10k–100k nodes typical

Without round boundaries:
- Tree could hit 50+ levels (exploring round transitions + scoring logic)
- Exponential blowup in search time
- Forced to use very shallow depth, killing strength

#### 4. Matches Human Mental Model
Humans naturally think in rounds:
> "If I place here, I score X and block that line. Next round I'll have access to Y tiles..."

Round boundaries align with how positions are intuitive and how features should generalize.

---

## Depth Configuration: Source-Adaptive

AlphaBeta uses `(depth, threshold)` where:
```python
sources = sum(tile counts across all colors and factories)
if sources > threshold:
    search_depth = depth
else:
    search_depth = sources  # search as deep as moves remain
```

**Rationale:**
- Early round: 50+ tiles across 5 colors → use fixed depth (1–3)
- Late round: 5–10 tiles → search full remainder of round (forces depth up to sources)
- Ensures thorough evaluation in tight endgames where one move locks a row

**Difficulty presets:**

| Difficulty | depth | threshold | Win Rate |
|---|---|---|---|
| easy | 1 | 4 | ~20% vs hard |
| medium | 2 | 6 | ~50% vs hard |
| hard | 3 | 8 | ≈ promotion bar |
| extreme | 4 | 10 | ~75% vs hard |

---

## How AlphaBeta Trains Value Nets

### Supervised Learning Loop

1. **Generate games** — Play AlphaBeta vs AlphaBeta (or weaker opponent)
   - Low temperature (0.1–0.3) to get deterministic, high-quality moves
   - Both players use same depth config (or opponent uses shallower)
   - Collect full game transcript

2. **Extract (state, outcome) pairs**
   - Per move: snapshot game state
   - Outcome: final score differential (my_score - opp_score) / 50
   - Label: (state, outcome) → train value net to predict outcome

3. **Train supervised network**
   - Input: 168-value player encoding [[encoding_strategy.md]]
   - Output: predicted value (real number, ±1 range approx)
   - Loss: MSE between predicted and actual outcome

4. **Replace heuristic** — In next iteration:
   - Old AlphaBeta used earned-score heuristic
   - New AlphaBeta uses learned net for leaf evaluation
   - Stronger value function → better search decisions

5. **Iterate**
   - New net-based AlphaBeta plays stronger
   - Generates better training games
   - Net improves again
   - Repeat until win rate plateaus

---

## Stochastic Move Selection

Despite deterministic tree search, AlphaBeta uses stochastic move selection at root:

```python
move_scores = [alphabeta(child) for child in root.children]
move_probs = softmax(move_scores / temperature)
selected_move = sample(move_probs)
```

**Two temperatures:**
- **Exploration temp (0.3):** Early/mid game — explores diverse moves
- **End-of-round temp (1.0):** Late round — higher entropy, less peaky

**Why stochastic?**
- Deterministic always picks best move → same games every run (if seed fixed)
- Stochastic samples around best → diverse games → diverse training data
- Still reproducible (controlled by seed)

---

## Comparison: AlphaBeta vs Other Agents

| Agent | Strength | Why |
|---|---|---|
| Random | Baseline | Uniform over legal moves |
| Greedy | ~49% | Color-conditional heuristic, no lookahead |
| Cautious | ~47% | Avoids floor, no depth |
| Minimax | ~50% | Depth-limited (1–2), no heuristic |
| **AlphaBeta** | **~55–60%** | **Depth 2–3 + earned-score heuristic** |
| AlphaZero (8z) | ~45–50% | Converged after 8 phases, never beat AB |

AlphaBeta's earned-score heuristic is better than anything AlphaZero's network learned from self-play.

---

## Practical Details

### Game Recording
Each game records:
- Moves (in compact string format)
- Scores per player after each move
- Earned score per player (to extract outcome)
- Seed (for reproducibility)

### Batch Processing
Games are generated in parallel via multiprocessing:
- Each worker plays independent game (deterministic given seed)
- Batch size: 10–50 games per iteration (TBD, tuned in Phase 1)
- Training runs on accumulated buffer

### Convergence Criteria
Iteration stops when:
- New net-based AlphaBeta win rate vs heuristic AB plateaus (3–5 iterations expected)
- Or max iterations reached (10 limit)
- Or performance regresses (net overfit to training distribution)

---

## Known Limitations

1. **Azul is luck-heavy** — Even perfect play has large win-rate variance (55–60% is not "solved")
2. **Factory seed variance** — Same position can have vastly different values next round depending on bag refill. Round boundaries help but don't eliminate this.
3. **Limited game diversity** — Both players using AlphaBeta = limited exploration. May need weak opponent or curriculum.

---

## Gotchas

### Critical: Advance & Earned Score Timing
```python
# WRONG: score_round() has happened, bonus folded into score
value = player.earned  # double-counts bonus

# CORRECT: read earned before advance
value = player.earned
game.advance(skip_setup=True)  # now score_round() has run
```

After `advance()`, `player.bonus` is non-zero but folded into `player.score`. Reading `earned` double-counts.

### Critical: Skip Setup on Tree Search
```python
# WRONG: next round is set up, changes position
game.advance()  # factories refilled, setup_round() called

# CORRECT: stop at round boundary
game.advance(skip_setup=True)  # round scored, next player selected, no factories
```

Tree evaluation must stay at round boundaries or value becomes meaningless.

---

## References

- [[master_plan.md]] — High-level vision
- [[encoding_strategy.md]] — State encoding (168 values)
- [[phase_0_refactor.md]] — Current Player refactoring work

---

**Status:** AlphaBeta is stable and production-ready. Ready to integrate into supervised training loop in Phase 1.
