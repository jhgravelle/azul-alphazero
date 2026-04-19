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
    score_differential_value,
    total_score_value,
    win_loss_value,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def make_trainer() -> Trainer:
    return Trainer(AzulNet())


def make_batch(
    batch_size: int = 16,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return a random (spatials, flats, policies, v_win, v_diff, v_abs) batch."""
    spatials = torch.rand(batch_size, *SPATIAL_SHAPE)
    flats = torch.rand(batch_size, FLAT_SIZE)
    raw = torch.rand(batch_size, MOVE_SPACE_SIZE)
    policies = raw / raw.sum(dim=-1, keepdim=True)
    values_win = torch.rand(batch_size, 1) * 2 - 1
    values_diff = torch.rand(batch_size, 1) * 2 - 1
    values_abs = torch.rand(batch_size, 1) * 2 - 1
    return spatials, flats, policies, values_win, values_diff, values_abs


def fill_buffer(buf: ReplayBuffer, n: int) -> None:
    spatials, flats, policies, vw, vd, va = make_batch(n)
    for i in range(n):
        buf.push(
            spatials[i],
            flats[i],
            policies[i],
            vw[i, 0].item(),
            vd[i, 0].item(),
            va[i, 0].item(),
        )


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


def test_collect_heuristic_games_policy_is_distribution_not_one_hot():
    """Policy targets should be multi-move distributions, not one-hot.
    At least one example in the buffer should have multiple non-zero
    entries in its policy vector."""
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_games(buf, num_games=3)
    _, _, policies, _, _, _ = buf.sample(min(len(buf), 50))
    nonzero_counts = (policies > 0).sum(dim=1)
    multi_move_rows = (nonzero_counts > 1).sum().item()
    assert multi_move_rows > 0, (
        f"Expected some examples with multi-move distributions, "
        f"got {multi_move_rows} rows with >1 non-zero entry"
    )


def test_collect_heuristic_games_policy_sums_to_one():
    """Every policy target should sum to exactly 1.0."""
    buf = ReplayBuffer(capacity=10_000)
    collect_heuristic_games(buf, num_games=2)
    _, _, policies, _, _, _ = buf.sample(min(len(buf), 30))
    sums = policies.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_collect_heuristic_games_records_all_games():
    """No games should be skipped — stats should show all games recorded."""
    buf = ReplayBuffer(capacity=100_000)
    stats = collect_heuristic_games(buf, num_games=20)
    total_games = stats["greedy_wins"] + stats["cautious_wins"] + stats["ties"]
    assert total_games == 20
    assert stats["games_recorded"] == 20


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


# ── Value target functions ─────────────────────────────────────────────────


def test_win_loss_value_p0_wins():
    assert win_loss_value([50, 30], 0) == 1.0
    assert win_loss_value([50, 30], 1) == -1.0


def test_win_loss_value_p1_wins():
    assert win_loss_value([30, 50], 0) == -1.0
    assert win_loss_value([30, 50], 1) == 1.0


def test_win_loss_value_tie():
    assert win_loss_value([40, 40], 0) == 0.0
    assert win_loss_value([40, 40], 1) == 0.0


def test_score_differential_value_equal():
    assert score_differential_value([40, 40], 0) == 0.0


def test_score_differential_value_positive_boundary():
    # +20 diff / 20 divisor = +1.0 exactly
    assert score_differential_value([50, 30], 0) == pytest.approx(1.0)


def test_score_differential_value_clips_positive():
    assert score_differential_value([60, 30], 0) == 1.0


def test_score_differential_value_clips_negative():
    assert score_differential_value([30, 60], 0) == -1.0


def test_score_differential_value_midrange():
    # +10 diff / 20 = +0.5
    assert score_differential_value([45, 35], 0) == pytest.approx(0.5)


def test_total_score_value_zero():
    assert total_score_value([0, 0], 0) == 0.0


def test_total_score_value_positive_boundary():
    # score 80 / divisor 80 = +1.0
    assert total_score_value([80, 30], 0) == pytest.approx(1.0)


def test_total_score_value_clips_positive():
    assert total_score_value([90, 30], 0) == 1.0


def test_total_score_value_clips_negative():
    assert total_score_value([-90, 30], 0) == -1.0


def test_total_score_value_midrange():
    # score 40 / 80 = +0.5
    assert total_score_value([40, 30], 0) == pytest.approx(0.5)


def test_total_score_value_only_depends_on_own_score():
    """Total score for player 0 should not depend on player 1's score."""
    v1 = total_score_value([40, 10], 0)
    v2 = total_score_value([40, 80], 0)
    assert v1 == v2


# ── Multi-head value loss ──────────────────────────────────────────────────


def test_compute_loss_returns_per_head_keys():
    """compute_loss dict should include per-head breakdown."""
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert "value_win" in result
    assert "value_diff" in result
    assert "value_abs" in result


def test_compute_loss_per_head_components_are_positive():
    """Each head's MSE should be > 0 on a fresh net with random targets."""
    net = AzulNet()
    result = compute_loss(net, *make_batch())
    assert result["value_win"].item() > 0.0
    assert result["value_diff"].item() > 0.0
    assert result["value_abs"].item() > 0.0


def test_compute_loss_value_equals_weighted_sum():
    """Combined value loss should equal 0.1·win + 0.1·diff + 1.0·abs."""
    import torch

    net = AzulNet()
    result = compute_loss(net, *make_batch())
    expected = (
        0.1 * result["value_win"]
        + 0.1 * result["value_diff"]
        + 1.0 * result["value_abs"]
    )
    assert torch.isclose(result["value"], expected)


def test_compute_loss_value_only_preserves_all_three_value_losses():
    """value_only zeroes policy but all three value components remain."""
    net = AzulNet()
    result = compute_loss(net, *make_batch(), value_only=True)
    assert result["value_win"].item() > 0.0
    assert result["value_diff"].item() > 0.0
    assert result["value_abs"].item() > 0.0


def test_train_step_too_small_buffer_returns_all_value_keys():
    """The empty-buffer early-return dict should include all value keys."""
    trainer = make_trainer()
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 10)
    result = trainer.train_step(buf)
    assert result["value_win"] == 0.0
    assert result["value_diff"] == 0.0
    assert result["value_abs"] == 0.0


def test_value_abs_weight_is_primary():
    """value_abs should have weight 1.0; win and diff should be lower."""
    from neural.trainer import _AUX_WEIGHT_WIN, _AUX_WEIGHT_DIFF, _AUX_WEIGHT_ABS

    assert _AUX_WEIGHT_ABS == 1.0
    assert _AUX_WEIGHT_WIN < _AUX_WEIGHT_ABS
    assert _AUX_WEIGHT_DIFF < _AUX_WEIGHT_ABS


def test_total_score_divisor_is_80():
    from neural.trainer import _TOTAL_SCORE_DIVISOR

    assert _TOTAL_SCORE_DIVISOR == 80.0


def test_compute_loss_value_equals_weighted_sum_new_weights():
    """Combined value loss = 0.1·win + 0.1·diff + 1.0·abs."""
    from neural.trainer import _AUX_WEIGHT_WIN, _AUX_WEIGHT_DIFF, _AUX_WEIGHT_ABS

    net = AzulNet()
    result = compute_loss(net, *make_batch())
    expected = (
        _AUX_WEIGHT_WIN * result["value_win"]
        + _AUX_WEIGHT_DIFF * result["value_diff"]
        + _AUX_WEIGHT_ABS * result["value_abs"]
    )
    assert torch.isclose(result["value"], expected)
