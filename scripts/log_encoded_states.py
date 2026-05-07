# scripts/log_encoded_states.py
"""Play one seeded game (AlphaBeta easy vs AlphaBeta medium) and log the
encoded state, policy distribution, and value targets for each turn to a file.

Uses _play_game and _history_to_records from trainer.py directly, so the log
reflects exactly what hits the replay buffer.

Usage:
    $env:PYTHONPATH = "."
    python -m scripts.log_encoded_states
"""

import torch

from agents.alphabeta import AlphaBetaAgent
from agents.base import Agent
from engine.game import CENTER, FLOOR, Game, Move
from neural.encoder import (
    decode_move,
    flat_policy_to_3head_targets,
    format_encoding,
)
from neural.trainer import _history_to_records, _play_game

OUTPUT_PATH = "encoded_states.txt"
SEED = 42

COLOR_LABELS = ["B", "Y", "R", "K", "W"]
DEST_LABELS = ["Line1", "Line2", "Line3", "Line4", "Line5", "Floor"]
SOURCE_LABELS = ["Center", "Factory"]


def _move_label(move: Move) -> str:
    """Return a short human-readable description of a move."""
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
    player_index: int,
    value_win: float,
    value_diff: float,
    value_abs: float,
    policy_valid: bool,
) -> str:
    """Format the four policy columns plus value targets as a single block."""
    # ── Top 5 moves + others ──────────────────────────────────────────────
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

    # ── Marginals ─────────────────────────────────────────────────────────
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

    # ── Value targets ─────────────────────────────────────────────────────
    value_lines = [
        f"{value_win:+.3f} win",
        f"{value_diff:+.3f} diff",
        f"{value_abs:+.3f} abs",
    ]

    # ── Combine columns ───────────────────────────────────────────────────
    rows = []
    for row_index in range(6):
        move_cell = move_col_lines[row_index] if row_index < len(move_col_lines) else ""
        color_cell = color_lines[row_index] if row_index < len(color_lines) else ""
        dest_cell = dest_lines[row_index] if row_index < len(dest_lines) else ""
        source_cell = source_lines[row_index] if row_index < len(source_lines) else ""
        value_cell = value_lines[row_index] if row_index < len(value_lines) else ""
        rows.append(
            f"{move_cell:<28}  {color_cell:<7}  {dest_cell:<12}  "
            f"{source_cell:<16}  {value_cell}"
        )

    return "\n".join(rows)


def main() -> None:
    alphabeta_easy = AlphaBetaAgent(depth=1, threshold=4)
    alphabeta_medium = AlphaBetaAgent(depth=1, threshold=4)
    agents: list[Agent] = [alphabeta_easy, alphabeta_medium]
    agent_names = ["AlphaBeta Easy", "AlphaBeta Medium"]

    game = Game(seed=SEED)
    game.setup_round()

    history, scores = _play_game(game, agents)
    records = _history_to_records(history, scores)

    lines: list[str] = []
    turn_number = 0
    round_number = 1
    lines.append(_round_header(round_number))
    lines.append("")

    for player_index, encoding_list, policy_list, vw, vd, va, policy_valid in records:
        encoding = torch.tensor(encoding_list, dtype=torch.float32)
        policy_vec = torch.tensor(policy_list, dtype=torch.float32)

        is_boundary = not policy_valid
        lines.append(format_encoding(encoding))
        lines.append("")
        lines.append(
            _format_policy_and_values(
                policy_vec, player_index, vw, vd, va, policy_valid
            )
        )
        lines.append("")

        if is_boundary:
            lines.append("=== END OF ROUND (pre-scoring, post-next_player) ===")
            lines.append("")
            round_number += 1
            lines.append(_round_header(round_number))
            lines.append("")
        else:
            turn_number += 1
            lines.append(
                f"=== Turn {turn_number} | Player {player_index} "
                f"({agent_names[player_index]}) ==="
            )
            lines.append("")

    lines.append(f"Final scores: Player 0 = {scores[0]}  Player 1 = {scores[1]}")

    with open(OUTPUT_PATH, "w") as output_file:
        output_file.write("\n".join(lines))

    print(f"Wrote {turn_number} turns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
