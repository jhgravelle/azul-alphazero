# tests/test_train.py
"""Tests for the train.py training loop helpers."""

import pytest
from neural.model import AzulNet
from neural.replay import ReplayBuffer

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
    _, _, _, values = buf.sample(min(len(buf), 32))
    assert values.min() >= -1.0
    assert values.max() <= 1.0


@pytest.mark.slow
def test_evaluate_with_buf_policies_sum_to_one():
    from scripts.train import evaluate
    import torch

    net = AzulNet()
    buf = ReplayBuffer(capacity=10_000)
    evaluate(net, net, num_games=2, simulations=2, buf=buf)
    _, _, policies, _ = buf.sample(min(len(buf), 32))
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
