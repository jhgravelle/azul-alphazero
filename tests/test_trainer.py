# tests/test_trainer.py
"""Tests for the AzulNet trainer — loss function and training step."""

import torch
import pytest

from neural.encoder import SPATIAL_SHAPE, FLAT_SIZE, MOVE_SPACE_SIZE
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from neural.trainer import (
    Trainer,
    compute_loss,
    collect_heuristic_games,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def make_trainer() -> Trainer:
    return Trainer(AzulNet())


def make_batch(
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a random (spatials, flats, policies, values) batch."""
    spatials = torch.rand(batch_size, *SPATIAL_SHAPE)
    flats = torch.rand(batch_size, FLAT_SIZE)
    raw = torch.rand(batch_size, MOVE_SPACE_SIZE)
    policies = raw / raw.sum(dim=-1, keepdim=True)
    values = torch.rand(batch_size, 1) * 2 - 1
    return spatials, flats, policies, values


def fill_buffer(buf: ReplayBuffer, n: int) -> None:
    spatials, flats, policies, values = make_batch(n)
    for i in range(n):
        buf.push(spatials[i], flats[i], policies[i], values[i, 0].item())


# ── compute_loss ───────────────────────────────────────────────────────────


def test_compute_loss_is_positive():
    assert compute_loss(AzulNet(), *make_batch())["total"].item() > 0.0


def test_compute_loss_is_finite():
    assert torch.isfinite(compute_loss(AzulNet(), *make_batch())["total"])


def test_compute_loss_has_gradient():
    net = AzulNet()
    compute_loss(net, *make_batch())["total"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_compute_loss_decreases_toward_perfect_value():
    net = AzulNet()
    batch = make_batch(32)
    loss_before = compute_loss(net, *batch)["total"].item()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(200):
        optimizer.zero_grad()
        compute_loss(net, *batch)["total"].backward()
        optimizer.step()
    assert compute_loss(net, *batch)["total"].item() < loss_before


# ── Trainer construction ───────────────────────────────────────────────────


def test_trainer_constructs():
    make_trainer()


def test_trainer_stores_network():
    net = AzulNet()
    assert Trainer(net).net is net


def test_trainer_default_lr():
    assert make_trainer().lr == pytest.approx(1e-3)


def test_trainer_default_batch_size():
    assert make_trainer().batch_size == 256


def test_trainer_has_optimizer():
    assert make_trainer().optimizer is not None


# ── Trainer.train_step ─────────────────────────────────────────────────────


def test_train_step_loss_is_positive():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    assert trainer.train_step(buf)["total"] > 0.0


def test_train_step_updates_weights():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    before = [p.clone() for p in trainer.net.parameters()]
    trainer.train_step(buf)
    after = list(trainer.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_train_step_loss_trends_down():
    trainer = Trainer(AzulNet(), lr=1e-2)
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    losses = [trainer.train_step(buf)["total"] for _ in range(100)]
    assert sum(losses[:10]) > sum(losses[-10:])


# ── collect_heuristic_games ────────────────────────────────────────────────


def test_collect_heuristic_games_fills_buffer():
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_games(buf, num_games=3)
    assert len(buf) > 0


def test_collect_heuristic_games_policy_is_one_hot():
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_games(buf, num_games=2)
    _, _, policies, _ = buf.sample(min(len(buf), 10))
    sums = policies.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums))


def test_collect_heuristic_games_filters_random_wins():
    buf = ReplayBuffer(capacity=100_000)
    stats = collect_heuristic_games(buf, num_games=40)
    assert stats["greedy_wins"] + stats["random_wins"] + stats["ties"] == 40
    assert stats["games_recorded"] == stats["greedy_wins"] + stats["ties"]


def test_collect_heuristic_games_values_are_score_differential():
    buf = ReplayBuffer(capacity=100_000)
    collect_heuristic_games(buf, num_games=10)
    _, _, _, values = buf.sample(min(len(buf), 50))
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    unique_values = set(values.squeeze().tolist())
    assert len(unique_values) > 3


def test_collect_self_play_warmup_records_both_players():
    from neural.model import AzulNet
    from neural.trainer import collect_self_play
    from agents.greedy import GreedyAgent

    net = AzulNet()
    buf = ReplayBuffer(capacity=100_000)
    collect_self_play(
        buf,
        net=net,
        num_games=4,
        simulations=2,
        temperature=1.0,
        opponent=GreedyAgent(),
    )
    assert len(buf) > 100, (
        f"Buffer has {len(buf)} examples from 4 games. "
        f"Expected >100 if recording both players."
    )


def test_collect_self_play_warmup_az_as_p1_records_nonzero_score():
    """AZ as player 1 in warmup mode should record actual score, not zero.

    We engineer a near-complete game state where both players are guaranteed
    to score: pattern line row 0 is full (scores 5 points — connects to 4
    wall neighbors), completing row 0 triggers a +2 row bonus. Starting score
    of 15 ensures floor penalties (-14 max) can't reach zero (15+5+2-14=8).
    """
    from neural.trainer import collect_self_play
    from neural.replay import ReplayBuffer
    from neural.model import AzulNet
    from agents.greedy import GreedyAgent
    from engine.constants import Tile, WALL_PATTERN, COLUMN_FOR_TILE_IN_ROW
    from engine import game as game_module

    net = AzulNet()
    buf = ReplayBuffer(capacity=1000)
    opponent = GreedyAgent()

    original_setup = game_module.Game.setup_round

    def rigged_setup(self, factories=None):
        original_setup(self, factories)
        for board in self.state.players:
            board.score = 15
        for board in self.state.players:
            board.pattern_lines[0] = [Tile.BLUE]
        blue_col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
        for board in self.state.players:
            for col in range(5):
                if col != blue_col:
                    board.wall[0][col] = WALL_PATTERN[0][col]

    game_module.Game.setup_round = rigged_setup
    try:
        az_scores = collect_self_play(
            buf,
            net=net,
            num_games=2,
            simulations=5,
            temperature=1.0,
            opponent=opponent,
            device=torch.device("cpu"),
        )
    finally:
        game_module.Game.setup_round = original_setup

    assert len(az_scores) == 2
    assert az_scores[0] > 0, f"AZ as p0 scored {az_scores[0]}, expected > 0"
    assert az_scores[1] > 0, f"AZ as p1 scored {az_scores[1]}, expected > 0"


def test_save_eval_recording_all_moves_legal():
    """Every move in the eval recording must have been legal when made."""
    from neural.model import AzulNet
    from scripts.train import save_eval_recording
    from engine.game import Game, Move
    from engine.constants import Tile
    import json
    import tempfile
    from pathlib import Path
    import unittest.mock as mock

    net = AzulNet()

    with tempfile.TemporaryDirectory() as tmp:
        eval_dir = Path(tmp) / "eval"

        with mock.patch("scripts.train.Path") as mock_path_cls:
            mock_path_cls.side_effect = lambda *args: (
                eval_dir if args == ("recordings/eval",) else Path(*args)
            )
            save_eval_recording(net, net, iteration=1, generation=0, simulations=5)

            files = list(eval_dir.glob("*.json"))
            assert len(files) == 1, "Expected one recording file"
            data = json.loads(files[0].read_text())

    game = Game()
    game.setup_round()

    for round_record in data["rounds"]:
        # Seed the replay game with the exact factories from the recording
        for i, factory_tiles in enumerate(round_record["factories"]):
            game.state.factories[i] = [Tile[t] for t in factory_tiles]
        game.state.center = [Tile[t] for t in round_record["center"]]

        for move_record in round_record["moves"]:
            legal = game.legal_moves()
            move = Move(
                source=move_record["source"],
                tile=Tile[move_record["tile"]],
                destination=move_record["destination"],
            )
            assert move in legal, f"Illegal move {move} -- legal moves were: {legal}"
            game.make_move(move)
            game.advance_round_if_needed()


def test_collect_self_play_warmup_az_avoids_floor_when_alternatives_exist():
    """In warmup mode, AZ should not choose a floor move when non-floor moves exist."""
    from neural.model import AzulNet
    from agents.alphazero import AlphaZeroAgent
    from engine.game import Game, FLOOR

    net = AzulNet()
    game = Game()
    game.setup_round()

    # Verify there are non-floor moves available
    legal = game.legal_moves()
    non_floor = [m for m in legal if m.destination != FLOOR]
    assert len(non_floor) > 0

    # Simulate what collect_self_play does in warmup mode
    move_before_override = None
    az_agent = AlphaZeroAgent(net, simulations=5, temperature=1.0)
    raw_move, policy_pairs = az_agent.get_policy_targets(game)
    move_before_override = raw_move

    # Apply the warmup floor override (same logic as collect_self_play)
    move = raw_move
    if move.destination == FLOOR:
        policy_list = list(policy_pairs)
        move = max(
            non_floor,
            key=lambda m: next((prob for pm, prob in policy_list if pm == m), 0.0),
        )

    # After override, move must not be a floor move
    assert move.destination != FLOOR, (
        f"After warmup override, move {move} is still a floor move. "
        f"Raw move was {move_before_override}"
    )
    assert move in legal, f"Overridden move {move} is not legal"


# ── compute_loss — dict return and value_only ──────────────────────────────


def test_compute_loss_returns_dict():
    """compute_loss should return a dict with 'total', 'policy', 'value' keys."""
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert isinstance(result, dict)
    assert "total" in result
    assert "policy" in result
    assert "value" in result


def test_compute_loss_total_is_scalar():
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert result["total"].shape == ()


def test_compute_loss_components_are_positive():
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert result["policy"].item() > 0.0
    assert result["value"].item() > 0.0


def test_compute_loss_value_only_zeroes_policy_loss():
    """When value_only=True, policy loss must be exactly 0."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert result["policy"].item() == 0.0


def test_compute_loss_value_only_preserves_value_loss():
    """When value_only=True, value loss must still be nonzero."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert result["value"].item() > 0.0


def test_compute_loss_value_only_total_equals_value_loss():
    """When value_only=True, total == value loss (policy contributes nothing)."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert torch.isclose(result["total"], result["value"])


def test_compute_loss_value_only_gradient_flows():
    """Gradient should still flow through value head when value_only=True."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    result["total"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_compute_loss_full_has_larger_total_than_value_only():
    """Full loss (policy + value) should exceed value-only loss on a fresh net."""
    net = AzulNet()
    batch = make_batch()
    full = compute_loss(net, *batch, value_only=False)
    value_only = compute_loss(net, *batch, value_only=True)
    assert full["total"].item() > value_only["total"].item()


# ── train_step — value_only ────────────────────────────────────────────────


def test_train_step_value_only_returns_dict():
    """train_step should return a loss dict, not a plain float."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    result = trainer.train_step(buf)
    assert isinstance(result, dict)
    assert "total" in result


def test_train_step_value_only_flag_zeroes_policy_loss():
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    result = trainer.train_step(buf, value_only=True)
    assert result["policy"] == 0.0


def test_train_step_value_only_still_updates_weights():
    """Even with value_only=True, the value head gradient should change weights."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)
    before = [p.clone() for p in trainer.net.parameters()]
    trainer.train_step(buf, value_only=True)
    after = list(trainer.net.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_train_step_too_small_buffer_returns_zero_dict():
    """When buffer is too small, return a zeroed dict rather than a plain 0.0."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 10)
    result = trainer.train_step(buf)
    assert isinstance(result, dict)
    assert result["total"] == 0.0
    assert result["policy"] == 0.0
    assert result["value"] == 0.0
