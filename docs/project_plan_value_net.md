Azul AlphaZero → Value Net Learning — Project Plan (Updated)

Status: Phase redesign complete (pivot from AlphaZero to supervised value learning)
Last updated: 2026-05-09
Goal: Build a superhuman Azul bot via learned value function + AlphaBeta search


Vision Shift
Original goal: AlphaZero-style self-play training → strong bot
Reality: Neural policy head stagnated (gen 1–2). MCTS distributions near-random. Training signal corrupted.
New approach: Abandon policy learning. Use supervised value learning instead.

AlphaBeta search provides ground-truth position evaluation
Train neural value net to match AB evaluations
Integrate learned value into AB search
Iterate: each generation's net produces better supervision for the next

Expected outcome: 90%+ win rate vs AB hard. Superhuman Azul play.

Architecture: AlphaBeta + Learned Value
Classical AlphaBeta search with two evaluation options:

Hand-Coded Eval (baseline):
  value = player.earned_differential (current score advantage)
  
Learned Value (gen-0+):
  encoding = encode_game(position)
  value = value_net(encoding)  [trained on AB supervision]
  
Integration:
  AlphaBetaWithLearnedValue(depth=3, threshold=8, net_path="gen_0000.pt")
    ├─ Searches to depth 3 or until round boundary
    ├─ Uses net prediction at leaf nodes (instead of hand-coded eval)
    └─ Returns best move
Why this works:

AlphaBeta is fast, deterministic, correct
Hand-coded eval has constant signal but limited strategic understanding
Neural value net learns to refine evaluation (cleanness, column strategy, end-game dynamics)
Generational improvement: gen-1 net sees better AB evaluations, produces better supervision for gen-2


Feature Engineering (LOCKED)
Player-Specific Features (Per Player: 35 values)
Columns (7 values)

5 column completion ratios (per column, normalized: filled / 5)
1 max column completion
1 second-max column completion

Rows (8 values)

5 row completion ratios (per row, normalized: filled / capacity)
1 top row (highest completion)
1 second-top row
1 third-top row

Pattern Lines (15 values)

5 pattern line fill ratios (current tiles / capacity per row)
1 total pattern tiles (sum of all pattern line tiles / 15)
5 pattern line cleanness (1.0 if wall color not placed, 0.0 if blocked)
5 pattern lines will be empty (1.0 if won't complete this round, 0.0 if will)

Line Flexibility (5 values)

5 incomplete lines per color (max count capped at 3, normalized)

Subtotal: 35 values per player × 2 = 70 values

Game-Level Features (6 values)

1 I hold first-player token (1.0 / 0.0)
1 Opponent holds first-player token (1.0 / 0.0)
1 Can I end game this round (1.0 / 0.0)
1 Can opponent end game this round (1.0 / 0.0)
1 Rounds remaining (round_num / 6)
1 Padding/reserved

Subtotal: 6 values

Static State Features (35 values)
Tile Availability (5 values)

Per color: tiles_remaining / 20

Source Counts (5 values)

Per color: source_count / 5

Bag Contents (5 values)

Per color: count_in_bag / 20

Official Scores (5 values)

My official score / 100
Opponent official score / 100
Padding × 3

Earned Scores (5 values)

My earned score / 100 (can be negative: [-0.14, 2+])
Opponent earned score / 100 (can be negative)
Padding × 3

Floor Penalties (5 values)

My penalty / 100 (range: [-0.14, 0.0])
Opponent penalty / 100 (range: [-0.14, 0.0])
Padding × 3

Bonus Points (5 values)

My bonus / 100
Opponent bonus / 100
Padding × 3

Subtotal: 35 values

Total Encoding: 111 values
0–69:     Player features (35 per player × 2)
70–75:    Game-level features (6)
76–80:    Tile availability (5)
81–85:    Source counts (5)
86–90:    Bag contents (5)
91–95:    Official scores (5)
96–100:   Earned scores (5)
101–105:  Floor penalties (5)
106–110:  Bonus points (5)

Total: 111 values
Normalization Principles:

All divisors use 100 for human interpretability (0.66 = 66 points)
Scores, earned, bonuses: raw division by 100 (can exceed 1.0 in late game)
Penalties: raw division by 100 (range [-0.14, 0.0], negative values allowed)
Ratios: division by their max (0.0–1.0 range)
No post-processing, no shifts, no inversions
Network learns to adjust negative values as needed


Data Generation Strategy
Phase 0: Supervision Data (Days 1–2)
Algorithm: Game playthrough with exhaustive search at round boundaries
pythondef generate_supervision_data(supervisor_agent, num_examples, output_file=None, threshold=5):
    """
    Generate supervision examples from random games.
    
    For smoke test (gen-0): store in RAM (< 10k examples)
    For scaling (gen-1+): stream to disk (Option 3 hybrid approach)
    
    Args:
        supervisor_agent: AlphaBetaAgent or AlphaBetaWithLearnedValue
        num_examples: target number of training examples
        output_file: if provided, stream to disk; if None, keep in RAM
        threshold: exhaustive search when <= this many moves remain in round
    
    Returns:
        training_data: list of (encoding, value) tuples (if RAM mode)
        OR output_file path (if disk mode)
    """
    training_data = []  # RAM mode (gen-0 smoke test)
    examples_collected = 0
    
    for game_idx in range(num_games_estimate):
        game = Game(seed=random())
        
        while not game.is_game_over():
            # Play randomly until near round boundary
            while not game.is_round_over():
                moves = game.legal_moves()
                game.make_move(random.choice(moves))
                game.advance(skip_setup=True)
                
                # Check: within threshold of round end?
                remaining_moves = estimate_remaining_moves(game)
                if remaining_moves <= threshold:  # threshold=5 or 6
                    # Run exhaustive AB search to round boundary
                    ab = AlphaBetaAgent(depth=999, threshold=0)  # Force full search
                    ab_tree = ab.build_complete_tree(game)
                    
                    # Extract all visited nodes
                    for node in ab_tree.all_nodes():
                        if node.was_visited:  # Skip pruned branches
                            encoding = encode_game(node.game_state)
                            value = node.value
                            
                            training_data.append((encoding, value))
                            examples_collected += 1
                            
                            # For disk streaming (gen-1+):
                            # if examples_collected % 1000 == 0:
                            #     _flush_to_disk(training_data, output_file, append=True)
                            #     training_data = []
            
            # Round boundary reached; continue to next round
            game.advance()
        
        # Game over
        if examples_collected >= num_examples:
            break
    
    # Final flush for disk mode
    if output_file and training_data:
        _flush_to_disk(training_data, output_file, append=True)
    
    return training_data if not output_file else output_file
Key points:

Play full random games to natural termination
When ≤ 5 moves remain in a round: exhaustive AB search (depth=999, threshold=0)
AB's pruning is trusted: if a branch is pruned, assume generalization will also skip it
Collect all visited nodes from the AB tree
Transitive benefit: depth-4 search implicitly gives us depth-3, depth-2, depth-1 data

Storage modes:

Gen-0 (smoke test): RAM-only, ~7 MB for 10k examples
Gen-1+ (scaling): Stream to disk every 1000 examples, load via PyTorch DataLoader

Wall time estimate:

5k examples: 1–2 hours
10k examples: 4–6 hours (overnight acceptable)
20k examples: 12+ hours (full day)

Smoke test approach: Start with 5k–10k, evaluate win rate. Scale up for later generations if needed.

Training Pipeline: Value Net
Phase 1: Supervised Value Learning (Days 2–4)
Architecture:
pythonclass AzulValueNet(nn.Module):
    """
    Neural value network for Azul position evaluation.
    
    Architecture:
      input (111) → Linear (64) → ResBlock (64) → Dropout (0.1)
                 → value_head (64 → 32 → 1)
    
    Output: Single scalar value per position
      - Represents: (my_earned - opponent_earned) / 100.0
      - Range: approximately [-0.14, 3.0+]
    """
    
    def __init__(self, input_size=111, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            ResBlock(hidden_dim),
            nn.Dropout(dropout),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, encoding):
        """
        Args:
            encoding: (batch_size, 111) tensor
        
        Returns:
            value: (batch_size,) tensor of scalar values
        """
        trunk_out = self.trunk(encoding)
        value = self.value_head(trunk_out).squeeze(-1)
        return value
Training Loop:
pythondef train_value_net(train_loader, val_loader, max_epochs=200, 
                    early_stopping_patience=20, lr=1e-3, 
                    save_dir="checkpoints/gen_0000/", device='cuda'):
    """
    Supervised training: minimize MSE between net predictions and AB labels.
    
    Args:
        train_loader: DataLoader with (encoding, ab_value) pairs
        val_loader: DataLoader for validation
        max_epochs: maximum epochs (early stopping usually stops earlier)
        early_stopping_patience: stop if val_loss doesn't improve for N epochs
        lr: learning rate
        save_dir: where to save checkpoint and logs
        device: 'cuda' or 'cpu'
    
    Returns:
        best_checkpoint_path: path to model with best val_loss
    """
    net = AzulValueNet(input_size=111).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_log = []
    
    for epoch in range(max_epochs):
        # ========== Training Phase ==========
        net.train()
        train_loss = 0.0
        
        for batch_encoding, batch_values in train_loader:
            batch_encoding = batch_encoding.to(device)
            batch_values = batch_values.to(device)
            
            optimizer.zero_grad()
            pred = net(batch_encoding)
            loss = loss_fn(pred, batch_values)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # ========== Validation Phase ==========
        net.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_encoding, batch_values in val_loader:
                batch_encoding = batch_encoding.to(device)
                batch_values = batch_values.to(device)
                
                pred = net(batch_encoding)
                loss = loss_fn(pred, batch_values)
                val_loss += loss.item()
                val_mae += torch.abs(pred - batch_values).mean().item()
                
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(batch_values.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Compute Pearson correlation
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        val_correlation = np.corrcoef(val_preds, val_targets)[0, 1]
        if np.isnan(val_correlation):
            val_correlation = 0.0
        
        # ========== Logging ==========
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_correlation': val_correlation,
            'learning_rate': lr,
        }
        training_log.append(log_entry)
        
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"mae={val_mae:.4f} | corr={val_correlation:.3f}")
        
        # ========== Early Stopping ==========
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint = f"{save_dir}/best_checkpoint.pt"
            torch.save(net.state_dict(), best_checkpoint)
            print(f"  → New best val_loss={val_loss:.4f}, saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (patience exhausted)")
                break
    
    # ========== Save Summary ==========
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training log
    import csv
    with open(f"{save_dir}/training_log.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    
    # Save summary
    with open(f"{save_dir}/summary.txt", 'w') as f:
        f.write(f"Generation Training Summary\n")
        f.write(f"Best epoch: {training_log.index(min(training_log, key=lambda x: x['val_loss']))}\n")
        f.write(f"Best val_loss: {best_val_loss:.6f}\n")
        f.write(f"Final train_loss: {training_log[-1]['train_loss']:.6f}\n")
        f.write(f"Final val_correlation: {training_log[-1]['val_correlation']:.4f}\n")
        f.write(f"Total epochs: {len(training_log)}\n")
    
    # Generate plots
    plot_training_curves(training_log, save_dir)
    
    return best_checkpoint
Hyperparameters:

Input size: 111 (encoder size, locked)
Hidden dim: 64 (small, effective)
Dropout: 0.1 (mild regularization)
Learning rate: 1e-3 (standard for supervised regression)
Batch size: 128
Train/val split: 80/20
Max epochs: 200
Early stopping patience: 20 epochs

Logging (per epoch):

train_loss, val_loss, val_mae, val_correlation, learning_rate
Save to CSV: checkpoints/gen_NNNN/training_log.csv

Checkpointing:

Save best checkpoint at lowest val_loss
Save final summary: summary.txt (best_epoch, best_val_loss, final_metrics)

Visualization (post-training):

loss_curves.png — train vs val loss over epochs
metrics_curves.png — val MAE + val correlation
All generated by matplotlib, saved to checkpoints/gen_NNNN/


Evaluation & Generational Loop
Phase 2: Evaluation (Every generation)
Test 1: vs Hand-Coded AlphaBeta
pythondef evaluate_agents(agent_a, agent_b, num_mirror_pairs=100, verbose=True):
    """
    Run mirrored pair evaluation.
    
    For each pair:
      Game 1: agent_a (player 0) vs agent_b (player 1), bag_seed S
      Game 2: agent_b (player 0) vs agent_a (player 1), bag_seed S (sides swapped)
    
    This controls for first-player advantage and randomness.
    
    Args:
        agent_a: first agent
        agent_b: second agent
        num_mirror_pairs: number of mirrored pairs to play (2 games per pair)
        verbose: print progress
    
    Returns:
        win_rate: fraction of games won by agent_a (0.0 to 1.0)
    """
    wins = 0
    total_games = 0
    
    for pair_idx in range(num_mirror_pairs):
        bag_seed = random()
        
        # Game 1: A vs B (A is player 0)
        game1 = Game(seed=bag_seed)
        winner1_idx = play_game(game1, [agent_a, agent_b])
        if winner1_idx == 0:
            wins += 1
        total_games += 1
        
        # Game 2: B vs A (A is player 1, sides swapped)
        game2 = Game(seed=bag_seed)
        winner2_idx = play_game(game2, [agent_b, agent_a])
        if winner2_idx == 1:  # A won (player 1 in this game)
            wins += 1
        total_games += 1
        
        if verbose and (pair_idx + 1) % 10 == 0:
            print(f"Completed {pair_idx + 1}/{num_mirror_pairs} pairs "
                  f"({total_games} games, {wins}/{total_games} wins for A)")
    
    win_rate = wins / total_games
    return win_rate
Test 2: vs AlphaBeta Hard (Progression Tracking)
python# Every generation, also test against AB hard
win_rate_vs_hard = evaluate_agents(
    AlphaBetaWithLearnedValue(depth=3, net_path=f"gen_{generation:04d}.pt"),
    AlphaBetaAgent(depth=3, threshold=8),  # AB hard
    num_mirror_pairs=100,
    verbose=True
)
Parameters:

Format: 100 mirrored pairs (200 games, controls for luck)
Depth: configurable, defaults to same as supervision depth (3)
Win rate threshold: 55% to declare "success"

Phase 3: Generational Loop (Days 5–14)
pythondef generational_loop(max_generations=10):
    """
    Automated generational loop: supervise → train → eval → decide.
    
    Each generation:
      1. Generate supervision data using current best net (or hand-coded for gen-0)
      2. Train new net on supervision
      3. Evaluate new net vs baseline and vs AB hard
      4. Log results, decide if improvement warrants next generation
    
    Stopping criteria:
      - max_generations reached
      - < 2% improvement for 3 consecutive generations (plateau)
      - User keyboard interrupt (graceful stop)
    """
    win_rates = {}
    plateau_counter = 0
    
    for generation in range(max_generations):
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")
        
        # ========== Step 1: Generate Supervision ==========
        if generation == 0:
            supervisor = AlphaBetaAgent(depth=3, threshold=0)
            print(f"Supervisor: Hand-coded AlphaBeta (depth=3, full search to round end)")
        else:
            supervisor = AlphaBetaWithLearnedValue(
                depth=3,
                threshold=0,
                net_path=f"checkpoints/gen_{generation-1:04d}/best_checkpoint.pt"
            )
            print(f"Supervisor: gen_{generation-1:04d} value net (depth=3)")
        
        target_examples = 10000 if generation < 3 else 15000  # Scale up after smoke test
        print(f"Generating {target_examples} supervision examples...")
        
        try:
            train_data, val_data = generate_supervision_data(
                supervisor,
                num_examples=target_examples,
                output_file=None if generation == 0 else f"checkpoints/gen_{generation:04d}/train_data.pt",
                threshold=5,
            )
            print(f"Generated {len(train_data)} training examples")
        except KeyboardInterrupt:
            print(f"Interrupted during data generation. Stopping.")
            break
        
        # ========== Step 2: Train Value Net ==========
        print(f"Training gen_{generation:04d} value net...")
        
        try:
            best_checkpoint = train_value_net(
                train_data,
                val_data,
                max_epochs=200,
                early_stopping_patience=20,
                lr=1e-3,
                save_dir=f"checkpoints/gen_{generation:04d}/",
            )
            print(f"Training complete. Best checkpoint: {best_checkpoint}")
        except KeyboardInterrupt:
            print(f"Interrupted during training. Stopping.")
            break
        
        # ========== Step 3: Evaluate vs Baseline ==========
        print(f"Evaluating gen_{generation:04d} vs hand-coded AB...")
        
        try:
            win_rate_baseline = evaluate_agents(
                AlphaBetaWithLearnedValue(depth=3, net_path=best_checkpoint),
                AlphaBetaAgent(depth=3, threshold=8),
                num_mirror_pairs=100,
                verbose=True,
            )
            print(f"Gen {generation} vs hand-coded: {win_rate_baseline*100:.1f}%")
        except KeyboardInterrupt:
            print(f"Interrupted during evaluation. Stopping.")
            break
        
        # ========== Step 4: Evaluate vs AB Hard ==========
        print(f"Evaluating gen_{generation:04d} vs AB hard...")
        
        try:
            win_rate_hard = evaluate_agents(
                AlphaBetaWithLearnedValue(depth=3, net_path=best_checkpoint),
                AlphaBetaAgent(depth=3, threshold=8),
                num_mirror_pairs=100,
                verbose=True,
            )
            print(f"Gen {generation} vs AB hard: {win_rate_hard*100:.1f}%")
        except KeyboardInterrupt:
            print(f"Interrupted during evaluation. Stopping.")
            break
        
        win_rates[generation] = {
            'vs_baseline': win_rate_baseline,
            'vs_hard': win_rate_hard,
        }
        
        # ========== Step 5: Analyze Progress ==========
        if generation > 0:
            prev_rate = win_rates[generation-1]['vs_baseline']
            improvement = win_rate_baseline - prev_rate
            print(f"\nImprovement over gen {generation-1}: {improvement*100:+.1f}%")
            
            if improvement < 0.02:  # < 2% improvement
                plateau_counter += 1
                print(f"Plateau warning ({plateau_counter}/3)")
                if plateau_counter >= 3:
                    print(f"Plateau detected for 3 consecutive generations. Stopping.")
                    break
            else:
                plateau_counter = 0  # Reset if we see improvement
        
        # ========== Step 6: User Confirmation ==========
        print(f"\n{'='*60}")
        if generation < max_generations - 1:
            try:
                prompt = f"Proceed to gen {generation+1}? (y/n/skip): "
                response = input(prompt).lower().strip()
                if response == 'n':
                    print(f"Stopping at gen {generation}.")
                    break
                elif response == 'skip':
                    print(f"Skipping eval games for gen {generation+1}.")
                    # (could implement skipping here)
            except KeyboardInterrupt:
                print(f"\nInterrupted. Saving results up to gen {generation}...")
                break
    
    # ========== Summary ==========
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    for gen, rates in sorted(win_rates.items()):
        print(f"Gen {gen:2d}: baseline={rates['vs_baseline']*100:5.1f}%, "
              f"hard={rates['vs_hard']*100:5.1f}%")
    
    # Find best generation
    best_gen = max(win_rates.keys(), key=lambda g: win_rates[g]['vs_hard'])
    print(f"\nBest generation: Gen {best_gen} (vs AB hard: {win_rates[best_gen]['vs_hard']*100:.1f}%)")
    print(f"Checkpoint: checkpoints/gen_{best_gen:04d}/best_checkpoint.pt")
Key decisions:

Max generations: 10 (hard cap)
Data per generation:

Gen 0–2: 10k examples (speed, smoke test)
Gen 3+: 15k–20k examples (refinement)


Plateau detection: < 2% improvement vs previous gen, 3 times in a row
Keyboard interrupt: Graceful stop with Ctrl+C, saves results
Evaluation: Every generation, 100 mirror pairs vs baseline + vs hard


Web Dashboard: Training Orchestration
Phase 5b: Training Dashboard (Days 6–8, parallel to Phase 5)
Architecture:
Frontend (Vanilla JS):
  ├─ Single-page dashboard (training_dashboard.html)
  ├─ Live polling of /training/status (every 2 seconds)
  ├─ Real-time metric updates, plots, controls
  └─ Manual start/stop/pause for each phase

Backend (FastAPI):
  ├─ Extended api/main.py with /training/* endpoints
  ├─ File-based state persistence (pipeline_state.json)
  ├─ Serves plots and generation history
  └─ Handles start/stop/pause signals

Orchestrator (Background Worker):
  ├─ Watches pipeline state file
  ├─ Runs generation → training → eval sequence
  ├─ Updates state in real-time
  └─ Graceful signal handling
Storage: File-based (per generation)
checkpoints/
├─ pipeline_state.json           ← Current pipeline state
├─ training_settings.json        ← User settings
└─ gen_NNNN/
   ├─ status.json               ← Phase, progress, metrics
   ├─ best_checkpoint.pt
   ├─ training_log.csv
   ├─ loss_curves.png
   ├─ metrics_curves.png
   ├─ summary.txt
   └─ live_metrics.json         ← Rolling window
Frontend: frontend/training_dashboard.html
Single-page HTML5 dashboard with:

Status card: Current phase (idle/generating/training/evaluating), progress bar
Live metrics: Games collected, epoch, train loss, val loss, correlation
Control panel: Start/stop buttons for each phase (enabled based on phase)
Settings form: Target examples, depth, LR, batch size (editable, saved to backend)
Generation table: History of all generations with stats
Plots: Loss curves and metrics curves (auto-updated)

Key features:

Vanilla JS (no build tools needed)
Live polling of /training/status endpoint
Graceful error handling
Responsive layout

Backend Endpoints: Extended api/main.py
python@app.get("/training/status")
async def get_training_status() -> TrainingStatus:
    """Get current pipeline status, metrics, and generation history."""
    # Returns: phase, current_gen, progress, games_collected, epoch,
    #          train/val loss, correlation, generation list, latest plots

@app.post("/training/start_generation")
async def start_generation() -> Dict:
    """Signal pipeline to start data generation."""

@app.post("/training/stop_generation")
async def stop_generation() -> Dict:
    """Signal pipeline to stop data generation."""

@app.post("/training/start_training")
async def start_training() -> Dict:
    """Signal pipeline to start training."""

@app.post("/training/stop_training")
async def stop_training() -> Dict:
    """Signal pipeline to stop training."""

@app.post("/training/start_evaluation")
async def start_evaluation() -> Dict:
    """Signal pipeline to start evaluation."""

@app.get("/training/generation/{gen_id}/plot/{plot_name}")
async def get_generation_plot(gen_id: int, plot_name: str) -> FileResponse:
    """Serve plot image (loss_curves.png or metrics_curves.png)."""

@app.post("/training/settings")
async def save_training_settings(settings: TrainingSettings) -> Dict:
    """Save training hyperparameters."""

@app.get("/training/settings")
async def get_training_settings() -> TrainingSettings:
    """Fetch current settings."""
Orchestrator: scripts/training_dashboard_worker.py
Background process that:

Watches checkpoints/pipeline_state.json for control signals
Detects phase changes (idle → generating, etc.)
Runs appropriate script (generate_ab_supervision.py, train_value_net.py, evaluate_agents.py)
Updates status.json in real-time with progress
Handles graceful stop/pause on signal

Can be run as:
bashpython scripts/training_dashboard_worker.py
Or as a background daemon during main API startup.

Feature Extraction Architecture
Player Features
Player owns and maintains all player-specific features:
python# engine/player.py

class Player:
    def __init__(self, player_idx: int):
        # Board state
        self.wall: List[List[int]] = [[0] * 5 for _ in range(5)]
        self.pattern_lines: List[List[Tile]] = [[] for _ in range(5)]
        self.floor: List[Tile] = []
        
        # Scoring caches
        self.score = 0
        self.pending = 0
        self.penalty = 0
        self.bonus = 0
        
        # Feature caches (updated on state change)
        self._column_completion_ratios = np.array([0.0] * 5, dtype=np.float32)
        self._row_completion_ratios = np.array([0.0] * 5, dtype=np.float32)
        self._pattern_line_fill_ratios = np.array([0.0] * 5, dtype=np.float32)
        self._pattern_line_cleanness = np.array([0.0] * 5, dtype=np.float32)
        self._pattern_lines_will_be_empty = np.array([0.0] * 5, dtype=np.float32)
        self._incomplete_lines_per_color = np.array([0.0] * 5, dtype=np.float32)
        self._max_column_completion = 0.0
        self._second_max_column_completion = 0.0
        self._top_row = 0.0
        self._second_top_row = 0.0
        self._third_top_row = 0.0
        self._total_pattern_tiles = 0.0
    
    # Property access (read-only)
    @property
    def column_completion_ratios(self) -> np.ndarray:
        return self._column_completion_ratios.copy()
    
    # ... more properties ...
    
    # Feature updates (called when state changes)
    def _update_column_features(self):
        """Recalculate column-based features."""
        for col in range(5):
            filled = sum(self.wall[row][col] for row in range(5))
            self._column_completion_ratios[col] = filled / 5.0
        
        self._max_column_completion = float(np.max(self._column_completion_ratios))
        self._second_max_column_completion = float(
            np.partition(self._column_completion_ratios, -2)[-2]
        )
    
    # ... more update methods ...
    
    # Feature extraction
    def extract_features(self) -> np.ndarray:
        """
        Extract all player-specific features as a 35-value vector.
        
        Returns: 35-value array
          - Columns: 7 values
          - Rows: 8 values
          - Pattern lines: 15 values
          - Line flexibility: 5 values
        """
        features = []
        
        # Columns (7)
        features.extend(self._column_completion_ratios)  # 5
        features.append(self._max_column_completion)  # 1
        features.append(self._second_max_column_completion)  # 1
        
        # Rows (8)
        features.extend(self._row_completion_ratios)  # 5
        features.append(self._top_row)  # 1
        features.append(self._second_top_row)  # 1
        features.append(self._third_top_row)  # 1
        
        # Pattern lines (15)
        features.extend(self._pattern_line_fill_ratios)  # 5
        features.append(self._total_pattern_tiles)  # 1
        features.extend(self._pattern_line_cleanness)  # 5
        features.extend(self._pattern_lines_will_be_empty)  # 5
        
        # Line flexibility (5)
        features.extend(self._incomplete_lines_per_color)  # 5
        
        return np.array(features, dtype=np.float32)
Game-Level Features
Game assembles both players' features and game-level features:
python# engine/game.py

class Game:
    def extract_features(self) -> np.ndarray:
        """
        Extract all game features as a 111-value vector.
        
        Returns: 111-value array
          - Player 0 features: 35
          - Player 1 features: 35
          - Game-level features: 6
          - Tile availability: 5
          - Source counts: 5
          - Bag contents: 5
          - Official scores: 5
          - Earned scores: 5
          - Floor penalties: 5
          - Bonus points: 5
        """
        features = []
        idx = 0
        
        # Player features (70)
        features.extend(self.players[0].extract_features())  # 35
        features.extend(self.players[1].extract_features())  # 35
        
        # Game-level features (6)
        features.append(1.0 if self.first_player_idx == 0 else 0.0)
        features.append(1.0 if self.first_player_idx == 1 else 0.0)
        features.append(self._can_player_end_game(0))
        features.append(self._can_player_end_game(1))
        features.append((6 - self.round_num) / 6.0)
        features.append(0.0)  # Padding
        
        # Tile availability (5)
        availability = self.tile_availability()
        for color_idx in range(5):
            features.append(availability[color_idx][0] / 20.0)
        
        # Source counts (5)
        for color_idx in range(5):
            features.append(availability[color_idx][1] / 5.0)
        
        # Bag contents (5)
        bag_contents = [0] * 5
        for tile in self.bag:
            bag_contents[tile.tile] += 1
        for color_idx in range(5):
            features.append(bag_contents[color_idx] / 20.0)
        
        # Official scores (5)
        features.append(self.players[0].score / 100.0)
        features.append(self.players[1].score / 100.0)
        features.append(0.0)
        features.append(0.0)
        features.append(0.0)
        
        # Earned scores (5)
        features.append(self.players[0].earned / 100.0)
        features.append(self.players[1].earned / 100.0)
        features.append(0.0)
        features.append(0.0)
        features.append(0.0)
        
        # Floor penalties (5)
        features.append(self.players[0].penalty / 100.0)
        features.append(self.players[1].penalty / 100.0)
        features.append(0.0)
        features.append(0.0)
        features.append(0.0)
        
        # Bonus points (5)
        features.append(self.players[0].bonus / 100.0)
        features.append(self.players[1].bonus / 100.0)
        features.append(0.0)
        features.append(0.0)
        features.append(0.0)
        
        return np.array(features, dtype=np.float32)
Encoder
Encoder simply calls game.extract_features():
python# neural/encoder.py

def encode_game(game: Game) -> np.ndarray:
    """
    Encode game state as 111-value flat vector.
    
    All values raw (no post-processing, negatives allowed):
    - Ratios: [0, 1]
    - Scores: [0, 2+] (late-game values can exceed 1.0)
    - Penalties: [-0.14, 0.0]
    - Earned: [-0.14, 2+] (can be negative)
    
    Network learns to interpret all ranges appropriately.
    """
    return game.extract_features()

Repository Structure (Updated)
azul-alphazero/
├── agents/
│   ├── alphabeta.py
│   ├── alphabeta_learned.py         ← NEW: AlphaBetaWithLearnedValue
│   ├── base.py
│   └── ... (existing agents)
│
├── neural/
│   ├── encoder.py                  ← UPDATED: simple wrapper around game.extract_features()
│   ├── model.py                    ← UPDATED: AzulValueNet (input_size=111)
│   ├── trainer.py                  ← UPDATED: train_value_net() for supervised learning
│   ├── replay.py                   ← May be repurposed or unused
│   └── search_tree.py              ← Unchanged
│
├── engine/
│   ├── player.py                   ← UPDATED: extract_features(), feature caches
│   ├── game.py                     ← UPDATED: extract_features(), _can_player_end_game()
│   ├── constants.py
│   └── ... (existing files)
│
├── scripts/
│   ├── generate_ab_supervision.py   ← NEW: Game playthrough + exhaustive AB search
│   ├── train_value_net.py          ← NEW: Supervised training loop
│   ├── evaluate_agents.py          ← NEW: Mirrored pair evaluation
│   ├── train_value_generations.py  ← NEW: Generational loop orchestrator
│   ├── training_dashboard_worker.py ← NEW: Background orchestrator
│   └── ... (existing scripts)
│
├── frontend/
│   ├── game.html                   ← Existing game UI
│   ├── game.js
│   ├── render.js
│   ├── style.css
│   └── training_dashboard.html     ← NEW: Training dashboard
│
├── api/
│   ├── main.py                    ← UPDATED: add /training/* endpoints
│   └── schemas.py
│
├── checkpoints/
│   ├── pipeline_state.json         ← NEW: Current pipeline state
│   ├── training_settings.json      ← NEW: User settings
│   └── gen_NNNN/
│       ├── status.json
│       ├── best_checkpoint.pt
│       ├── training_log.csv
│       ├─ loss_curves.png
│       ├─ metrics_curves.png
│       └─ summary.txt
│
├── tests/
│   ├── neural/
│   │   ├── test_encoder.py         ← UPDATED: test 111-value encoder
│   │   ├── test_model.py           ← UPDATED: test AzulValueNet
│   │   └── test_trainer.py         ← UPDATED: test train_value_net()
│   ├── agents/
│   │   └── test_alphabeta_learned.py ← NEW
│   ├── engine/
│   │   └── test_features.py        ← NEW: test Player.extract_features()
│   ├── integration/
│   │   └── test_value_pipeline.py  ← NEW: end-to-end pipeline test
│   └── ...
│
├── docs/
│   ├── project_plan.md             ← THIS FILE
│   └── value_net_training.md       ← NEW: tutorial
│
└── ... (existing structure)

Development Phases
Phase 0: Feature Extraction Architecture (Days 1)
Estimated wall time: 3–4 hours

 Player feature implementation

 Add feature cache properties to Player
 Add _update_*_features() methods
 Add extract_features() method returning 35-value array
 Call update methods in place_tile(), score_placement(), etc.


 Game feature implementation

 Add extract_features() method returning 111-value array
 Add _can_player_end_game() helper
 Test full feature extraction pipeline


 Encoder simplification

 Update encode_game() to call game.extract_features()
 Remove old 125-value encoding logic
 Verify encoding size is exactly 111


 Testing

 Unit test: Player.extract_features() shape and range
 Unit test: Game.extract_features() shape and range
 Integration test: encode a game position, verify 111 values
 Spot-check feature values for correctness



Commit: feat(features): player-owned feature extraction, 111-value encoding

Phase 1: Data Generation (Days 1–2)
Estimated wall time: 6–8 hours + overnight

 Script implementation

 Create scripts/generate_ab_supervision.py
 Implement game playthrough loop
 Implement round-boundary detection
 Implement exhaustive AB search (depth=999, threshold=0)
 Node extraction from AB tree


 Smoke test

 Generate 1k examples (10 min), verify shape/quality
 Generate 10k examples (overnight run)
 Verify data distribution


 Documentation

 Comment on data generation algorithm
 Note wall time per example count



Expected output:

checkpoints/gen_0000/ directory with training data

Commit: feat(data): game playthrough with exhaustive AB search at round boundaries

Phase 2: Value Net Training (Days 2–4)
Estimated wall time: 4–6 hours + overnight

 Model implementation

 Implement AzulValueNet in neural/model.py (input_size=111)
 Ensure proper initialization


 Training implementation

 Implement train_value_net() in scripts/train_value_net.py
 Per-epoch logging (train_loss, val_loss, val_mae, val_correlation)
 Early stopping mechanism
 Checkpoint management


 Visualization

 Implement matplotlib loss curves
 Implement metrics curves
 Save plots to PNG


 Smoke test

 Train gen-0 net on 10k supervision examples (overnight)
 Verify training progress
 Expected: val_loss ~0.04–0.08, correlation > 0.90


 Testing

 Unit test: AzulValueNet output shape
 Unit test: train_value_net() completes without error
 Check loss curves save correctly



Expected output:

checkpoints/gen_0000/best_checkpoint.pt
checkpoints/gen_0000/training_log.csv
checkpoints/gen_0000/loss_curves.png, metrics_curves.png
checkpoints/gen_0000/summary.txt

Commit: feat(neural): AzulValueNet + supervised training pipeline

Phase 3: AlphaBeta Integration (Days 3–4)
Estimated wall time: 2–4 hours

 Agent implementation

 Implement AlphaBetaWithLearnedValue in agents/alphabeta_learned.py
 Load value net from checkpoint at construction
 Override _evaluate_position() to use net prediction
 Fallback to hand-coded eval if net unavailable


 Integration tests

 Unit test: agent constructs without error
 Unit test: agent makes valid moves
 Manual smoke test: play a few games


 Registration

 Add to agents/registry.py
 Update api/schemas.py with new agent type



Commit: feat(agents): AlphaBetaWithLearnedValue integration

Phase 4: Evaluation (Days 4–5)
Estimated wall time: 3–5 hours per generation

 Evaluation script

 Create scripts/evaluate_agents.py
 Implement evaluate_agents() with 100 mirrored pairs
 Support configurable depth
 Verbose output


 Gen-0 evaluation

 Evaluate gen-0 vs hand-coded AB at depth 3
 100 mirror pairs (200 games)
 Expected: 55%+ win rate if learning happened
 Log results


 Decision

 If > 55%, proceed to generational loop
 If < 55%, debug or pivot



Expected output:

checkpoints/gen_0000/eval_results.csv
checkpoints/gen_0000/eval_summary.txt

Commit: feat(eval): mirrored pair evaluation framework

Phase 5: Generational Loop (Days 5–14)
Estimated wall time: 60+ hours (spread over multiple nights)

 CLI orchestrator

 Create scripts/train_value_generations.py
 Implement generational loop
 Plateau detection
 Graceful keyboard interrupt handling


 Automated progression

 Each generation uses previous net as supervisor
 Scale data: gen 0–2 = 10k, gen 3+ = 15k–20k
 Track win rates per generation
 Stop on plateau or max_generations


 Results tracking

 Save generation progression table
 Final recommendation


 Testing

 Integration test: full gen-0 → gen-1 cycle
 Verify state persistence



Expected progression:

Gen-0: 55–65% vs baseline
Gen-1: 60–70% vs baseline
Gen-2: 65–75% vs baseline
Gen-3+: diminishing returns, plateau by gen-5–7

Commit: feat(train): generational loop with plateau detection

Phase 6: UI Integration (Days 5–8, parallel to Phase 5)
Estimated wall time: 4–6 hours

 Inspector updates

 Add value net selector to inspector UI
 Show value net eval alongside AB eval
 Smooth transitions when switching nets


 Game UI updates

 Add agent selector for game
 Allow playing against different generations


 API updates

 Expose value net agents via /agents endpoint
 Load best generation automatically


 Tests

 Agent selector UI doesn't crash
 Games work against learned-value agents



Commit: feat(ui): value net agent selector in inspector and game

Phase 7: Web Dashboard (Days 6–8, parallel to Phase 5)
Estimated wall time: 8–10 hours

 Frontend implementation (frontend/training_dashboard.html)

 Status card, live metrics, control panel
 Settings form, generation table
 Plot display
 Vanilla JS, no build tools


 Backend implementation (extend api/main.py)

 /training/* endpoints
 File-based state (pipeline_state.json)


 Orchestrator (scripts/training_dashboard_worker.py)

 Watch state file for signals
 Run generation/train/eval scripts
 Update status in real-time


 Integration

 Start dashboard worker on API startup (optional)
 Dashboard polls correctly
 Buttons enable/disable based on phase
 Plots display after training/eval


 Testing

 Manual: open dashboard, start generation
 Watch metrics update in real-time
 Stop and resume generation
 Plots generate and display



Commit: feat(dashboard): training pipeline web dashboard

Phase 8: Logging & Visualization (Days 7–9, parallel)
Estimated wall time: 4–6 hours

 Structured logs

 Generational progression table (CSV)
 Master log: checkpoints/generation_log.csv


 Plots

 Win rate per generation
 Best val_loss per generation
 Wall time breakdown per generation
 All generations' loss curves on same plot (optional)


 Summary

 "Best generation: Gen X with Y% win rate"
 Recommendation for deployment



Commit: feat(logging): generation comparison plots and experiment registry

Phase 9: Documentation (Days 8–9)
Estimated wall time: 4–6 hours

 Update project_plan.md

 Replace old AlphaZero sections
 Clarify architecture shift
 Update backlog


 New document: docs/value_net_training.md

 Tutorial: "How to train a value net from scratch"
 Step-by-step: phases 0–5
 Dashboard walkthrough
 Expected timelines and win rates


 Inline code comments

 Why supervised learning
 Why exhaustive search at round boundaries
 Why mirrored pairs for evaluation


 README section

 "Value Net Training" section
 Quick start commands



Commit: docs: value net training guide and architecture overview

Phase 10: Testing & Polish (Days 9–11)
Estimated wall time: 6–8 hours

 Full test suite

 Update encoder tests (111 values)
 AzulValueNet forward pass
 train_value_net() flow
 AlphaBetaWithLearnedValue correctness
 Evaluation mirrored pairs
 Dashboard API endpoints
 Player feature extraction


 Edge cases

 No examples generated → error handling
 Training diverges → early stopping works
 Eval all draws → statistics handling
 Checkpoint missing → graceful fallback


 CI/CD

 All tests pass locally
 GitHub Actions workflow updated
 No regressions on existing game functionality


 Cleanup

 Remove old AlphaZero code (TBD)
 Or keep as reference
 Consolidate duplicate code
 Remove debug prints



Commit: test: full coverage for value net pipeline

Success Criteria
MilestoneMetricTargetGen-0 trainingVal correlation on test set> 0.90Gen-0 evaluationWin rate vs hand-coded AB≥ 55%Gen-2 evaluationWin rate vs hand-coded AB≥ 65%Gen-3+ evaluationWin rate vs hand-coded AB≥ 75%Final generationWin rate vs AB hard≥ 85%Superhuman targetWin rate vs AB hard≥ 90%

Timeline Summary
PhaseDaysWall TimeStatusPhase 0: Features1~3–4 hoursReady to implementPhase 1: Data Gen1–2~6–8 hours + overnightReadyPhase 2: Training2–4~4–6 hours + overnightReadyPhase 3: Integration3–4~2–4 hoursReadyPhase 4: Evaluation4–5~3–5 hoursReadyPhase 5: Gen Loop5–14~60+ hours (spread over nights)ReadyPhase 6: UI Integration5–8~4–6 hoursReadyPhase 7: Dashboard6–8~8–10 hoursReadyPhase 8: Logging7–9~4–6 hoursReadyPhase 9: Documentation8–9~4–6 hoursReadyPhase 10: Testing & Polish9–11~6–8 hoursReadyTotal~2 weeks~100 hours computation, 40 hours manualReady to start

Hard-Won Lessons (Updated)
Building on experience from AlphaZero phase:

Supervised learning is simpler than RL. Ground-truth labels (AB evaluations) are reliable. No credit assignment horror.
Value learning >> policy learning for Azul. Move selection (policy) is already solved by AlphaBeta. What's missing is position evaluation (value).
Mirrored pairs eliminate luck. Azul has high variance. Always use mirrored pairs for evaluation.
Early stopping is your friend. Don't manually set epochs. Let validation loss guide training.
Generational improvement is compounding but plateaus fast. Expect 5–10% improvement per gen early, then 1–2% later. Plateau is normal.
Exhaustive search at round boundaries is the right data. When factories empty and pattern lines are committed, the value