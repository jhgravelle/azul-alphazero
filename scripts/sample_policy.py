# scripts/sample_policy.py
"""Sample net policy and value head outputs across many random game positions.

Generates N random game states (by default at turn 1 of round 1) and prints
the top-k policy moves and all value heads for each. Useful for diagnosing
value head calibration — e.g. checking whether value_win is consistently
near 0.0 on empty boards (correct) or near ±1.0 (overfit/biased).

Usage:
    python -m scripts.sample_policy
    python -m scripts.sample_policy --checkpoint checkpoints/latest.pt
    python -m scripts.sample_policy --samples 50 --top-k 3 --turn 1
    python -m scripts.sample_policy --turn 5 --samples 20
"""

from __future__ import annotations

import argparse
import statistics
import torch
import torch.nn.functional as F

from engine.game import Game, FLOOR
from engine.constants import Tile
from agents.alphabeta import AlphaBetaAgent
from neural.model import AzulNet
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE


# ── Helpers ────────────────────────────────────────────────────────────────


def _move_label(move) -> str:
    from engine.game import CENTER

    source = "CTR" if move.source == CENTER else f"F{move.source + 1}"
    color = move.tile.name[:3]
    dest = "floor" if move.destination == FLOOR else f"row{move.destination + 1}"
    return f"{source} {color}->{dest}"


def _net_forward(
    net: AzulNet,
    game: Game,
    legal: list,
) -> tuple[list[tuple], float, float, float]:
    """Return (policy_dist, value_win, value_diff, value_abs)."""
    spatial, flat = encode_state(game)
    spatial = spatial.unsqueeze(0)
    flat = flat.unsqueeze(0)
    net.eval()
    with torch.no_grad():
        logits, vw, vd, va = net(spatial, flat)
    value_win = vw.item()
    value_diff = vd.item()
    value_abs = va.item()
    logits = logits.squeeze(0)
    mask = torch.full((MOVE_SPACE_SIZE,), float("-inf"))
    for move in legal:
        idx = encode_move(move, game)
        mask[idx] = logits[idx]
    probs = F.softmax(mask, dim=0)
    policy_dist = [(move, probs[encode_move(move, game)].item()) for move in legal]
    return policy_dist, value_win, value_diff, value_abs


def _advance_to_turn(game: Game, turn: int, agent: AlphaBetaAgent) -> bool:
    """Play (turn - 1) moves using AlphaBeta to reach the desired turn.

    Returns False if the game ended before reaching the target turn.
    """
    for _ in range(turn - 1):
        if game.is_game_over():
            return False
        legal = game.legal_moves()
        if not legal:
            return False
        move = agent.choose_move(game)
        game.make_move(move)
        game.advance()
        if game.is_game_over():
            return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample net policy and value heads across random game states"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint to load (default: fresh random net)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="number of random game states to sample (default 100)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="top-k policy moves to show per state (default 3)",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=1,
        help="which turn of round 1 to sample (default 1 = game start)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print only the summary statistics, not individual game states",
    )
    args = parser.parse_args()

    net = AzulNet()
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded: {args.checkpoint}")
    else:
        print("Using fresh (random) net")

    # Use depth-1 AlphaBeta to advance to the target turn quickly
    agent = AlphaBetaAgent(depths=(1, 1, 1), thresholds=(20, 10))

    value_wins: list[float] = []
    value_diffs: list[float] = []
    value_abss: list[float] = []
    top1_is_floor: list[bool] = []
    top1_probs: list[float] = []
    n_legal_moves: list[int] = []

    succeeded = 0

    for sample_idx in range(args.samples):
        game = Game()
        game.setup_round()

        if args.turn > 1:
            ok = _advance_to_turn(game, args.turn, agent)
            if not ok:
                continue

        legal = game.legal_moves()
        if not legal:
            continue

        policy_dist, vw, vd, va = _net_forward(net, game, legal)
        policy_sorted = sorted(policy_dist, key=lambda x: x[1], reverse=True)

        value_wins.append(vw)
        value_diffs.append(vd)
        value_abss.append(va)
        top1_is_floor.append(policy_sorted[0][0].destination == FLOOR)
        top1_probs.append(policy_sorted[0][1])
        n_legal_moves.append(len(legal))
        succeeded += 1

        if not args.summary_only:
            cur = game.state.current_player
            print(
                f"\n[{sample_idx + 1:03d}] turn={args.turn} player={cur} "
                f"legal={len(legal)}  "
                f"v_win={vw:+.4f}  v_diff={vd:+.4f}  v_abs={va:+.4f}"
            )
            for rank, (move, prob) in enumerate(policy_sorted[: args.top_k], 1):
                floor_tag = " [FLOOR]" if move.destination == FLOOR else ""
                print(f"  top{rank}: {_move_label(move):<22} {prob:.4f}{floor_tag}")

    # ── Summary ────────────────────────────────────────────────────────────
    if succeeded == 0:
        print("No valid samples generated.")
        return

    print("\n" + "=" * 60)
    print(f"Summary over {succeeded} samples (turn {args.turn})")
    print("=" * 60)

    def _stats(values: list[float], label: str) -> None:
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        mn = min(values)
        mx = max(values)
        print(
            f"  {label:<20}  mean={mean:+.4f}  std={stdev:.4f}  "
            f"min={mn:+.4f}  max={mx:+.4f}"
        )

    print("\nValue heads:")
    _stats(value_wins, "value_win")
    _stats(value_diffs, "value_diff")
    _stats(value_abss, "value_abs")

    print("\nPolicy head:")
    floor_pct = sum(top1_is_floor) / len(top1_is_floor) * 100
    mean_top1_prob = statistics.mean(top1_probs)
    mean_legal = statistics.mean(n_legal_moves)
    uniform_prob = 1.0 / mean_legal
    print(f"  top-1 is floor move:   {floor_pct:.1f}% of samples")
    print(
        f"  avg top-1 probability: {mean_top1_prob:.4f}  "
        f"(uniform would be {uniform_prob:.4f})"
    )
    print(f"  avg legal moves:       {mean_legal:.1f}")

    print("\nDiagnosis:")
    # value_win on an empty board should be near 0 (50/50 game)
    mean_vw = statistics.mean(value_wins)
    std_vw = statistics.stdev(value_wins) if len(value_wins) > 1 else 0.0
    if abs(mean_vw) > 0.3:
        print(
            f"  *** value_win mean {mean_vw:+.4f} is far from 0 — "
            f"possible bias in training data (player perspective skew) ***"
        )
    elif std_vw > 0.4:
        print(
            f"  *** value_win std {std_vw:.4f} is high — "
            f"net is sensitive to factory configuration noise, likely overfit ***"
        )
    else:
        print(f"  value_win looks calibrated (mean={mean_vw:+.4f}, std={std_vw:.4f})")

    mean_vd = statistics.mean(value_diffs)
    std_vd = statistics.stdev(value_diffs) if len(value_diffs) > 1 else 0.0
    if abs(mean_vd) > 0.2:
        print(
            f"  *** value_diff mean {mean_vd:+.4f} is far from 0 — "
            f"systematic bias in score differential targets ***"
        )
    elif std_vd > 0.3:
        print(
            f"  *** value_diff std {std_vd:.4f} is high — "
            f"overfit to specific factory patterns ***"
        )
    else:
        print(f"  value_diff looks calibrated (mean={mean_vd:+.4f}, std={std_vd:.4f})")

    if floor_pct > 20:
        print(
            f"  *** top-1 policy move is a floor move {floor_pct:.1f}% of the time — "
            f"policy head prefers floor ***"
        )
    else:
        print(f"  policy head floor preference: {floor_pct:.1f}% (acceptable)")


if __name__ == "__main__":
    main()
