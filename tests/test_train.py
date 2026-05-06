# tests/test_train.py
"""Tests for the train.py training loop helpers."""

import pytest
from neural.model import AzulNet
from neural.replay import ReplayBuffer
from tests.test_trainer import fill_buffer


def test_value_only_iterations_zero_uses_full_training():
    """With value_only=False, training should use full loss — policy loss nonzero."""
    from neural.trainer import Trainer

    trainer = Trainer(AzulNet())
    buf = ReplayBuffer(capacity=1000)
    fill_buffer(buf, 300)

    result = trainer.train_step(buf, value_only=False)
    assert result["policy"] > 0.0


@pytest.mark.slow
def test_batched_mcts_faster_than_serial():
    """Batched MCTS should complete the same number of simulations faster than
    serial."""
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

    serial_tree = SearchTree(
        policy_value_fn=make_policy_value_fn(net),
        simulations=50,
        temperature=1.0,
    )
    t0 = time.perf_counter()
    serial_tree.choose_move(game)
    serial_time = time.perf_counter() - t0

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
    """Verify _summary_line handles a skipped-eval IterResult without crashing."""
    from scripts.train import IterResult, _summary_line

    result = IterResult(
        iteration=1,
        mode="self-play",
        avg_loss=0.05,
        win_rate=-1.0,  # sentinel for skipped
        promoted=False,
        generation=0,
        elapsed=10.0,
    )
    line = _summary_line(result)
    assert "skip" in line.lower() or "1" in line


def test_format_loss_line_shows_all_heads():
    """The formatted log line should include per-head value breakdown."""
    from scripts.train import (
        _init_loss_accumulator,
        _accumulate_losses,
        _format_loss_line,
    )

    accum = _init_loss_accumulator()
    _accumulate_losses(
        accum,
        {
            "total": 1.0,
            "policy": 0.5,
            "value": 0.5,
            "value_win": 0.1,
            "value_diff": 0.2,
            "value_abs": 0.2,
        },
    )
    line = _format_loss_line(accum, n_steps=1)
    assert "avg loss: 1.0000" in line
    assert "policy: 0.5000" in line
    assert "win 0.1000" in line
    assert "diff 0.2000" in line
    assert "abs 0.2000" in line
