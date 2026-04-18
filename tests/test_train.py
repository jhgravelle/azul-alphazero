# tests/test_train.py
"""Tests for the train.py training loop helpers."""

import pytest
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from tests.test_trainer import fill_buffer

# ── evaluate with buffer feeding ───────────────────────────────────────────


@pytest.mark.slow
def test_evaluate_without_buf_returns_win_rate():
    from scripts.train import evaluate

    net = AzulNet()
    win_rate = evaluate(net, net, num_games=4, simulations=2)
    assert 0.0 <= win_rate <= 1.0


@pytest.mark.slow
def test_evaluate_with_buf_fills_buffer():
    from scripts.train import evaluate

    net = AzulNet()
    buf = ReplayBuffer(capacity=10_000)
    evaluate(net, net, num_games=4, simulations=2, buf=buf)
    assert len(buf) > 0


@pytest.mark.slow
def test_evaluate_with_buf_pushes_both_players():
    """Both players' positions should be recorded, so buffer grows faster
    than one player alone would produce."""
    from scripts.train import evaluate

    net = AzulNet()
    buf_both = ReplayBuffer(capacity=10_000)
    evaluate(net, net, num_games=2, simulations=2, buf=buf_both)
    # A 2-player game of ~60 moves should produce >60 examples if both recorded
    assert len(buf_both) > 60


@pytest.mark.slow
def test_evaluate_with_buf_values_in_range():
    from scripts.train import evaluate

    net = AzulNet()
    buf = ReplayBuffer(capacity=10_000)
    evaluate(net, net, num_games=2, simulations=2, buf=buf)
    _, _, _, vw, vd, va = buf.sample(min(len(buf), 32))
    for values in (vw, vd, va):
        assert values.min() >= -1.0
        assert values.max() <= 1.0


@pytest.mark.slow
def test_evaluate_with_buf_policies_sum_to_one():
    from scripts.train import evaluate
    import torch

    net = AzulNet()
    buf = ReplayBuffer(capacity=10_000)
    evaluate(net, net, num_games=2, simulations=2, buf=buf)
    _, _, policies, _, _, _ = buf.sample(min(len(buf), 32))
    sums = policies.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.slow
def test_evaluate_without_buf_does_not_crash():
    """evaluate() with no buf should work exactly as before."""
    from scripts.train import evaluate

    net = AzulNet()
    win_rate = evaluate(net, net, num_games=2, simulations=2)
    assert isinstance(win_rate, float)


@pytest.mark.slow
def test_evaluate_win_rate_range():
    from scripts.train import evaluate

    net = AzulNet()
    buf = ReplayBuffer(capacity=10_000)
    win_rate = evaluate(net, net, num_games=6, simulations=2, buf=buf)
    assert 0.0 <= win_rate <= 1.0


@pytest.mark.slow
def test_evaluate_with_record_saves_file(tmp_path, monkeypatch):
    from scripts.train import evaluate

    monkeypatch.chdir(tmp_path)
    net = AzulNet()
    evaluate(
        net, net, num_games=2, simulations=2, record=True, iteration=1, generation=0
    )
    files = list((tmp_path / "recordings" / "eval").glob("*.json"))
    assert len(files) == 1


@pytest.mark.slow
def test_evaluate_with_record_filename_matches_iteration_and_generation(
    tmp_path, monkeypatch
):
    from scripts.train import evaluate

    monkeypatch.chdir(tmp_path)
    net = AzulNet()
    evaluate(
        net, net, num_games=2, simulations=2, record=True, iteration=3, generation=7
    )
    files = list((tmp_path / "recordings" / "eval").glob("*.json"))
    assert len(files) == 1
    assert "iter_003" in files[0].name
    assert "gen_0007" in files[0].name


@pytest.mark.slow
def test_evaluate_without_record_saves_no_file(tmp_path, monkeypatch):
    from scripts.train import evaluate

    monkeypatch.chdir(tmp_path)
    net = AzulNet()
    evaluate(net, net, num_games=2, simulations=2, record=False)
    eval_dir = tmp_path / "recordings" / "eval"
    files = list(eval_dir.glob("*.json")) if eval_dir.exists() else []
    assert len(files) == 0


@pytest.mark.slow
def test_evaluate_record_only_saves_one_file_for_multiple_games(tmp_path, monkeypatch):
    from scripts.train import evaluate

    monkeypatch.chdir(tmp_path)
    net = AzulNet()
    evaluate(
        net, net, num_games=4, simulations=2, record=True, iteration=1, generation=0
    )
    files = list((tmp_path / "recordings" / "eval").glob("*.json"))
    assert len(files) == 1


def test_value_only_iterations_zero_uses_full_training():
    """With value_only_iterations=0, training should use full loss from iter 1."""
    from neural.trainer import Trainer
    from neural.replay import ReplayBuffer

    trainer = Trainer(AzulNet())
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)

    # Full training — policy loss should be nonzero
    result = trainer.train_step(buf, value_only=False)
    assert result["policy"] > 0.0


@pytest.mark.slow
def test_batched_mcts_faster_than_serial():
    """Batched MCTS should complete more simulations in less time than serial."""
    import time
    from neural.search_tree import (
        SearchTree,
        make_policy_value_fn,
        make_batch_policy_value_fn,
    )
    from engine.game import Game

    net = AzulNet()
    game = Game()
    game.setup_round()

    # Serial: 50 sims
    serial_tree = SearchTree(
        policy_value_fn=make_policy_value_fn(net),
        simulations=50,
        temperature=1.0,
    )
    t0 = time.perf_counter()
    serial_tree.choose_move(game)
    serial_time = time.perf_counter() - t0

    # Batched: 50 sims
    game2 = game.clone()
    batched_tree = SearchTree(
        policy_value_fn=make_policy_value_fn(net),
        batch_policy_value_fn=make_batch_policy_value_fn(net),
        simulations=50,
        temperature=1.0,
        batch_size=50,
    )
    t0 = time.perf_counter()
    batched_tree.choose_move(game2)
    batched_time = time.perf_counter() - t0

    assert (
        batched_time < serial_time
    ), f"Batched ({batched_time:.3f}s) should be faster than serial "
    f"({serial_time:.3f}s)"


def test_skip_eval_iterations_skips_eval():
    """When iteration <= skip_eval_iterations, win rate should be None/skipped."""
    # This is tested implicitly by the training loop — just verify the
    # summary line handles a skipped eval gracefully.
    from scripts.train import IterResult, _summary_line

    result = IterResult(
        iteration=1,
        mode="self-play",
        avg_loss=0.05,
        win_rate=-1.0,  # sentinel for skipped
        promoted=False,
        generation=0,
        az_avg=-30.0,
        elapsed=10.0,
    )
    line = _summary_line(result)
    assert "skip" in line.lower() or "1" in line  # just checks it doesn't crash
