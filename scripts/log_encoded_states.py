# scripts/log_encoded_states.py
"""Play one seeded game (AlphaZero vs AlphaZero) and log the
encoded state, policy distribution, value targets, and net predictions for
each turn to a file.

Uses _play_game and _history_to_records from trainer.py directly, so the log
reflects exactly what hits the replay buffer.

Usage:
    $env:PYTHONPATH = "."
    python -m scripts.log_encoded_states
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from agents.base import Agent
from engine.game import CENTER, FLOOR, Game, Move
from neural.encoder import (
    decode_move,
    flat_policy_to_3head_targets,
    format_encoding,
)
from neural.model import AzulNet
from neural.trainer import _history_to_records

OUTPUT_PATH = "encoded_states.txt"
CHECKPOINT_PATH = Path("checkpoints/latest.pt")
SEED = 42

COLOR_LABELS = ["B", "Y", "R", "K", "W"]
DEST_LABELS = ["Line1", "Line2", "Line3", "Line4", "Line5", "Floor"]
SOURCE_LABELS = ["Center", "Factory"]


def _move_label(move: Move) -> str:
    source = "Center" if move.source == CENTER else f"F{move.source}"
    destination = (
        "Floor" if move.destination == FLOOR else f"Line{move.destination + 1}"
    )
    return f"{source} {move.tile.name}->{destination}"


def _round_header(round_number: int) -> str:
    bar = "=" * 60
    return f"{bar}\n=== ROUND {round_number:<51}===\n{bar}"


def _format_policy_and_values(
    policy_vec: torch.Tensor,
    value_win: float,
    value_diff: float,
    value_abs: float,
    policy_valid: bool,
) -> str:
    """Format AB policy targets and value targets."""
    nonzero_indices = policy_vec.nonzero(as_tuple=True)[0].tolist()
    move_probs = sorted(
        [(policy_vec[idx].item(), idx) for idx in nonzero_indices],
        key=lambda pair: pair[0],
        reverse=True,
    )
    top_five = move_probs[:5]
    others_probability = sum(p for p, _ in move_probs[5:])
    others_count = len(move_probs) - 5

    move_col_lines = []
    for probability, idx in top_five:
        move = decode_move(idx, None)  # type: ignore[arg-type]
        move_col_lines.append(f"{probability:.3f}  {_move_label(move)}")
    move_col_lines.append(f"{others_probability:.3f}  Others ({others_count})")

    if policy_valid:
        src_targets, tile_targets, dst_targets = flat_policy_to_3head_targets(
            policy_vec.unsqueeze(0)
        )
        color_lines = [
            f"{p:.2f} {label}"
            for label, p in zip(COLOR_LABELS, tile_targets[0].tolist())
        ]
        dest_lines = [
            f"{p:.2f} {label:<5}"
            for label, p in zip(DEST_LABELS, dst_targets[0].tolist())
        ]
        source_lines = [
            f"{p:.2f} {label}"
            for label, p in zip(SOURCE_LABELS, src_targets[0].tolist())
        ]
    else:
        color_lines = ["     " for _ in COLOR_LABELS]
        dest_lines = ["          " for _ in DEST_LABELS]
        source_lines = ["              " for _ in SOURCE_LABELS]

    value_lines = [
        f"{value_win:+.3f} win",
        f"{value_diff:+.3f} diff",
        f"{value_abs:+.3f} abs",
    ]

    rows = ["AB targets:"]
    for row_index in range(6):
        move_cell = move_col_lines[row_index] if row_index < len(move_col_lines) else ""
        color_cell = color_lines[row_index] if row_index < len(color_lines) else ""
        dest_cell = dest_lines[row_index] if row_index < len(dest_lines) else ""
        source_cell = source_lines[row_index] if row_index < len(source_lines) else ""
        value_cell = value_lines[row_index] if row_index < len(value_lines) else ""
        rows.append(
            f"  {move_cell:<28}  {color_cell:<7}  {dest_cell:<12}  "
            f"{source_cell:<16}  {value_cell}"
        )
    return "\n".join(rows)


def _format_net_predictions(net: AzulNet, encoding: torch.Tensor) -> str:
    """Run encoding through net and format policy head outputs vs AB targets."""
    with torch.no_grad():
        (src_logits, tile_logits, dst_logits), value_win, value_diff, value_abs = net(
            encoding.unsqueeze(0)
        )

    src_probs = F.softmax(src_logits[0], dim=0)
    tile_probs = F.softmax(tile_logits[0], dim=0)
    dst_probs = F.softmax(dst_logits[0], dim=0)

    color_lines = [
        f"{p:.2f} {label}" for label, p in zip(COLOR_LABELS, tile_probs.tolist())
    ]
    dest_lines = [
        f"{p:.2f} {label:<5}" for label, p in zip(DEST_LABELS, dst_probs.tolist())
    ]
    source_lines = [
        f"{p:.2f} {label}" for label, p in zip(SOURCE_LABELS, src_probs.tolist())
    ]
    value_lines = [
        f"{value_win.item():+.3f} win",
        f"{value_diff.item():+.3f} diff",
        f"{value_abs.item():+.3f} abs",
    ]

    rows = ["NET predictions:"]
    for row_index in range(6):
        color_cell = color_lines[row_index] if row_index < len(color_lines) else ""
        dest_cell = dest_lines[row_index] if row_index < len(dest_lines) else ""
        source_cell = source_lines[row_index] if row_index < len(source_lines) else ""
        value_cell = value_lines[row_index] if row_index < len(value_lines) else ""
        rows.append(
            f"  {'':28}  {color_cell:<7}  {dest_cell:<12}  "
            f"{source_cell:<16}  {value_cell}"
        )
    return "\n".join(rows)


def _load_net() -> AzulNet | None:
    if not CHECKPOINT_PATH.exists():
        return None
    net = AzulNet()
    net.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    net.eval()
    return net


def main() -> None:
    net = _load_net()

    if net is None:
        print("No checkpoint found — cannot run AZ vs AZ")
        return

    from agents.alphazero import AlphaZeroAgent

    agent_0 = AlphaZeroAgent(net, simulations=50, temperature=0.0)
    agent_1 = AlphaZeroAgent(net, simulations=50, temperature=0.0)
    agents: list[Agent] = [agent_0, agent_1]
    agent_names = ["AlphaZero", "AlphaZero"]

    game = Game(seed=SEED)
    game.setup_round()

    # Capture moves alongside history by replaying manually
    from agents.alphazero import AlphaZeroAgent as AZ

    chosen_moves: list[Move | None] = []
    history_raw: list[tuple] = []
    prev_round = game.round

    agent_0.reset_tree(game)
    agent_1.reset_tree(game)

    while True:
        if not game.legal_moves():
            break
        current = game.current_player_index
        agent = agents[current]
        encoding = __import__("neural.encoder", fromlist=["encode_state"]).encode_state(
            game
        )

        if isinstance(agent, AZ):
            move, policy_pairs = agent.get_policy_targets(game)
        else:
            move = agent.choose_move(game)
            policy_pairs = agent.policy_distribution(game)

        from neural.encoder import encode_move, MOVE_SPACE_SIZE

        policy_vec = torch.zeros(MOVE_SPACE_SIZE)
        for m, prob in policy_pairs:
            policy_vec[encode_move(m, game)] = prob

        history_raw.append((current, encoding, policy_vec, True, move))
        chosen_moves.append(move)

        game.make_move(move)

        if game.is_round_over() and not game.is_game_over():
            boundary_game = game.clone()
            boundary_game.next_player()
            from neural.encoder import encode_state

            boundary_enc = encode_state(boundary_game)
            null_policy = torch.zeros(MOVE_SPACE_SIZE)
            history_raw.append(
                (
                    boundary_game.current_player_index,
                    boundary_enc,
                    null_policy,
                    False,
                    None,
                )
            )
            chosen_moves.append(None)

        game.advance()
        if game.is_game_over():
            break

        if game.round != prev_round:
            for a in agents:
                if isinstance(a, AZ):
                    a.reset_tree(game)
            prev_round = game.round
        else:
            for a in agents:
                if isinstance(a, AZ):
                    a.advance(move)

    scores = [p.score for p in game.players]

    history_for_records = [
        (pi, enc, pv, valid) for pi, enc, pv, valid, _ in history_raw
    ]
    records = _history_to_records(history_for_records, scores)

    lines: list[str] = [f"Checkpoint: {CHECKPOINT_PATH}", ""]
    turn_number = 0
    round_number = 1
    lines.append(_round_header(round_number))
    lines.append("")

    for (player_index, encoding_list, policy_list, vw, vd, va, policy_valid), (
        _,
        _,
        _,
        _,
        chosen_move,
    ) in zip(records, history_raw):
        encoding = torch.tensor(encoding_list, dtype=torch.float32)
        policy_vec = torch.tensor(policy_list, dtype=torch.float32)

        is_boundary = not policy_valid
        lines.append(format_encoding(encoding))
        lines.append("")
        lines.append(_format_policy_and_values(policy_vec, vw, vd, va, policy_valid))
        lines.append("")

        if net is not None:
            lines.append(_format_net_predictions(net, encoding))
            lines.append("")

        if is_boundary:
            lines.append("=== END OF ROUND (pre-scoring, post-next_player) ===")
            lines.append("")
            round_number += 1
            lines.append(_round_header(round_number))
            lines.append("")
        else:
            turn_number += 1
            move_str = (
                f"  chose: {_move_label(chosen_move)}"
                if chosen_move is not None
                else ""
            )
            lines.append(
                f"=== Turn {turn_number} | Player {player_index} "
                f"({agent_names[player_index]}){move_str} ==="
            )
            lines.append("")

    lines.append(f"Final scores: Player 0 = {scores[0]}  Player 1 = {scores[1]}")

    with open(OUTPUT_PATH, "w") as output_file:
        output_file.write("\n".join(lines))

    print(f"Wrote {turn_number} turns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
