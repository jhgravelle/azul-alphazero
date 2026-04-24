# scripts/inspect_policy.py
"""Inspect AlphaBeta policy distribution vs net policy output for one game.

Plays one game between two AlphaBeta agents, and at each move prints:
  - The game state (factories, center, both boards)
  - AlphaBeta's scored policy distribution (what goes into the buffer)
  - The net's current policy output for that state
  - KL divergence between the two distributions

Usage:
    python -m scripts.inspect_policy
    python -m scripts.inspect_policy --checkpoint checkpoints/latest.pt
    python -m scripts.inspect_policy --moves 5 --top-k 8
"""

from __future__ import annotations

import argparse
import math
import torch
import torch.nn.functional as F

from engine.game import Game, CENTER, FLOOR
from engine.constants import COLOR_TILES, WALL_PATTERN, COLUMN_FOR_TILE_IN_ROW, Tile
from agents.alphabeta import AlphaBetaAgent
from neural.model import AzulNet
from neural.encoder import encode_state, encode_move, MOVE_SPACE_SIZE


# ── Formatting helpers ─────────────────────────────────────────────────────


def _tile_char(tile: Tile | None) -> str:
    if tile is None:
        return "."
    return tile.name[0]


def _format_wall_row(wall_row: list, pattern_row: list, pattern_line: list) -> str:
    """One row: wall cells | pattern line fill."""
    wall_str = " ".join(_tile_char(cell) for cell in wall_row)
    if pattern_line:
        fill = " ".join(_tile_char(t) for t in pattern_line)
        color = pattern_line[0].name[:3]
        capacity = len(pattern_row)
        line_str = f"{fill} ({color} {len(pattern_line)}/{capacity})"
    else:
        line_str = "-"
    return f"  wall: {wall_str}  |  line: {line_str}"


def _format_board(board, label: str) -> str:
    lines = [f"{label}  score={board.score}"]
    for row in range(5):
        lines.append(
            _format_wall_row(
                board.wall[row],
                WALL_PATTERN[row],
                board.pattern_lines[row],
            )
        )
    floor_str = " ".join(_tile_char(t) for t in board.floor_line) or "-"
    lines.append(f"  floor: {floor_str}")
    return "\n".join(lines)


def _format_sources(game: Game) -> str:
    lines = []
    for f_idx, factory in enumerate(game.state.factories):
        if factory:
            tiles = " ".join(t.name[:3] for t in factory)
            lines.append(f"  F{f_idx + 1}: {tiles}")
    center_tiles = [t for t in game.state.center if t != Tile.FIRST_PLAYER]
    fp = " +FP" if Tile.FIRST_PLAYER in game.state.center else ""
    if center_tiles or fp:
        ctr = " ".join(t.name[:3] for t in center_tiles)
        lines.append(f"  CTR: {ctr}{fp}")
    return "\n".join(lines) if lines else "  (empty)"


def _move_label(move) -> str:
    source = "CTR" if move.source == CENTER else f"F{move.source + 1}"
    color = move.tile.name[:3]
    dest = "floor" if move.destination == FLOOR else f"row{move.destination + 1}"
    return f"{source} {color}->{dest}"


# ── Net policy and value helpers ──────────────────────────────────────────


def _net_forward(
    net: AzulNet,
    game: Game,
    legal: list,
) -> tuple[list[tuple], float, float, float]:
    """Run net forward pass, return policy dist and all three value heads.

    Returns:
        policy_dist:  [(move, prob)] for all legal moves
        value_win:    scalar in (-1, 1) — win/loss prediction
        value_diff:   scalar in (-1, 1) — score differential prediction
        value_abs:    scalar in (-1, 1) — absolute score prediction
    """
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


def _print_encoding(game: Game) -> None:
    """Print the flat encoding vector alongside expected values.

    Shows each flat feature with its encoded value, the value we expect
    from the game state, and whether they match. Helps catch perspective
    flips, wrong divisors, or stale state bugs.
    """
    from neural.encoder import (
        OFF_FACTORIES,
        OFF_CENTER,
        OFF_FP_CENTER,
        OFF_FP_MINE,
        OFF_MY_FLOOR,
        OFF_OPP_FLOOR,
        OFF_MY_SCORE,
        OFF_OPP_SCORE,
        OFF_SCORE_DELTA,
        OFF_BAG,
        OFF_DISCARD,
        OFF_ROUND,
        OFF_DISTINCT_PAIRS,
        FLAT_SIZE,
        NUM_COLORS,
        SCORE_DELTA_DIVISOR,
    )
    from engine.scoring import earned_score_unclamped
    from engine.constants import TILES_PER_COLOR, TILES_PER_FACTORY, Tile

    _, flat = encode_state(game)
    cur = game.state.current_player
    opp = 1 - cur
    my = game.state.players[cur]
    op = game.state.players[opp]

    my_score = earned_score_unclamped(my)
    opp_score = earned_score_unclamped(op)
    delta = max(-1.0, min(1.0, (my_score - opp_score) / SCORE_DELTA_DIVISOR))

    print(f"\nFlat encoding (current_player={cur}):")
    print(f"  {'Feature':<35} {'encoded':>8}  {'expected':>8}  match")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}  -----")

    def row(label, enc_val, exp_val, tol=1e-4):
        match = "OK" if abs(enc_val - exp_val) <= tol else "*** MISMATCH ***"
        print(f"  {label:<35} {enc_val:>8.4f}  {exp_val:>8.4f}  {match}")

    # First player token
    row(
        "fp_in_center",
        flat[OFF_FP_CENTER].item(),
        1.0 if Tile.FIRST_PLAYER in game.state.center else 0.0,
    )
    row(
        "fp_mine (player " + str(cur) + ")",
        flat[OFF_FP_MINE].item(),
        1.0 if Tile.FIRST_PLAYER in my.floor_line else 0.0,
    )

    # Floor lines
    row(f"my_floor (player {cur})", flat[OFF_MY_FLOOR].item(), len(my.floor_line) / 7)
    row(f"opp_floor (player {opp})", flat[OFF_OPP_FLOOR].item(), len(op.floor_line) / 7)

    # Scores
    row(f"my_score (player {cur}) /100", flat[OFF_MY_SCORE].item(), my_score / 100)
    row(f"opp_score (player {opp}) /100", flat[OFF_OPP_SCORE].item(), opp_score / 100)
    row(
        f"score_delta (my-opp)/{SCORE_DELTA_DIVISOR:.0f}",
        flat[OFF_SCORE_DELTA].item(),
        delta,
    )

    # Round and distinct pairs
    row(
        "round_progress (round-1)/5", flat[OFF_ROUND].item(), (game.state.round - 1) / 5
    )
    expected_pairs = game.count_distinct_source_color_pairs()
    row("distinct_pairs /10", flat[OFF_DISTINCT_PAIRS].item(), expected_pairs / 10)

    # Bag totals — spot check first color
    from engine.constants import COLOR_TILES

    for color_idx, color in enumerate(COLOR_TILES):
        enc = flat[OFF_BAG + color_idx].item()
        exp = game.state.bag.count(color) / TILES_PER_COLOR
        row(f"bag[{color.name[:3]}] /20", enc, exp)

    # Discard totals — spot check
    for color_idx, color in enumerate(COLOR_TILES):
        enc = flat[OFF_DISCARD + color_idx].item()
        exp = game.state.discard.count(color) / TILES_PER_COLOR
        row(f"discard[{color.name[:3]}] /20", enc, exp)

    # Factory spot check — just show nonzero entries
    print(f"\n  Factory encoding (nonzero entries):")
    for f_idx, factory in enumerate(game.state.factories):
        for color_idx, color in enumerate(COLOR_TILES):
            enc = flat[OFF_FACTORIES + f_idx * NUM_COLORS + color_idx].item()
            exp = factory.count(color) / TILES_PER_FACTORY
            if enc > 0 or exp > 0:
                row(f"  factory[{f_idx}][{color.name[:3]}] /4", enc, exp)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect AlphaBeta vs net policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to checkpoint to load (default: fresh random net)",
    )
    parser.add_argument(
        "--moves",
        type=int,
        default=10,
        help="number of moves to inspect before stopping (default 10)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="top-k moves to display per distribution (default 10)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="AlphaBeta search depth (default 1 for speed)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=200,
        help="MCTS simulations for the search probe per move (default 200)",
    )
    parser.add_argument(
        "--show-encoding",
        action="store_true",
        help="print the flat encoding vector alongside expected values for debugging",
    )
    args = parser.parse_args()

    net = AzulNet()
    if args.checkpoint:
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using fresh (random) net")

    agent = AlphaBetaAgent(
        depths=(args.depth, args.depth, args.depth),
        thresholds=(20, 10),
    )

    game = Game()
    game.setup_round()

    # ── Scoring diagnostic ─────────────────────────────────────────────────
    # Verify earned_score_unclamped delta for a row-0 (capacity 1) fill.
    # A move that fills row 0 should show a positive delta since
    # earned_score_unclamped includes pending placement points.
    from engine.scoring import earned_score_unclamped as _esu

    _legal = game.legal_moves()
    _row0_moves = [m for m in _legal if m.destination == 0]
    if _row0_moves:
        _before = _esu(game.state.players[0])
        _child = game.clone()
        _child.make_move(_row0_moves[0])
        _after = _esu(_child.state.players[0])
        _line = _child.state.players[0].pattern_lines[0]
        _wall = _child.state.players[0].wall[0]
        print(f"\nScoring diagnostic (row-0 fill):")
        print(f"  move:         {_move_label(_row0_moves[0])}")
        print(f"  pattern[0]:   {_line}  (capacity 1, full={len(_line) >= 1})")
        print(f"  wall[0]:      {_wall}")
        print(f"  floor:        {_child.state.players[0].floor_line}")
        print(f"  before:       {_before}")
        print(f"  after:        {_after}")
        print(f"  delta:        {_after - _before}")
    # ── End diagnostic ─────────────────────────────────────────────────────

    move_count = 0
    kl_values = []

    while not game.is_game_over() and move_count < args.moves:
        print("\n" + "=" * 70)
        print(
            f"Move {move_count + 1} | Round {game.state.round} | "
            f"Player {game.state.current_player}"
        )
        print("-" * 70)

        # Sources
        print("Sources:")
        print(_format_sources(game))

        # Both boards
        cur = game.state.current_player
        opp = 1 - cur
        print(_format_board(game.state.players[cur], f"Player {cur} (current):"))
        print(_format_board(game.state.players[opp], f"Player {opp} (opponent):"))

        if args.show_encoding:
            _print_encoding(game)

        # AlphaBeta policy
        move = agent.choose_move(game)
        ab_dist = agent.policy_distribution(game)
        ab_dist_sorted = sorted(ab_dist, key=lambda x: x[1], reverse=True)

        print(f"\nAlphaBeta policy (top {args.top_k})  [chosen: {_move_label(move)}]")
        print(f"  {'Move':<22} {'AB prob':>8}  {'score bar'}")
        for m, prob in ab_dist_sorted[: args.top_k]:
            bar = "#" * int(prob * 40)
            marker = (
                " <-- chosen"
                if (
                    m.source == move.source
                    and m.tile == move.tile
                    and m.destination == move.destination
                )
                else ""
            )
            print(f"  {_move_label(m):<22} {prob:>8.4f}  {bar}{marker}")

        remaining = len(ab_dist) - args.top_k
        if remaining > 0:
            tail_prob = sum(p for _, p in ab_dist_sorted[args.top_k :])
            print(f"  ... {remaining} more moves, combined prob={tail_prob:.4f}")

        # Net policy + value heads
        legal = game.legal_moves()
        net_dist, value_win, value_diff, value_abs = _net_forward(net, game, legal)
        net_dist_sorted = sorted(net_dist, key=lambda x: x[1], reverse=True)

        # Value head summary for current position
        puct_head = value_diff  # this is what PUCT uses in search_tree.py
        print(f"\nNet value heads (current position):")
        print(f"  value_win:  {value_win:+.4f}  (win/loss, range -1 to +1)")
        print(
            f"  value_diff: {value_diff:+.4f}  (score differential / 50)  <-- PUCT uses this"
        )
        print(f"  value_abs:  {value_abs:+.4f}  (absolute score / 100)")

        # Show value heads after specific moves for comparison
        floor_moves = [m for m in legal if m.destination == FLOOR]
        best_net_move = net_dist_sorted[0][0] if net_dist_sorted else None
        moves_to_probe = []
        if best_net_move is not None:
            moves_to_probe.append(("net top move", best_net_move))
        if floor_moves:
            moves_to_probe.append(("floor move", floor_moves[0]))

        if moves_to_probe:
            print(f"\n  Value heads after specific moves:")
            print(f"  {'Move':<28} {'v_win':>7} {'v_diff':>7} {'v_abs':>7}")
            for label, probe_move in moves_to_probe:
                probe_game = game.clone()
                probe_game.make_move(probe_move)
                probe_legal = probe_game.legal_moves()
                _, pw, pd, pa = _net_forward(net, probe_game, probe_legal or legal)
                print(
                    f"  {label+': '+_move_label(probe_move):<28} "
                    f"{pw:>+7.4f} {pd:>+7.4f} {pa:>+7.4f}"
                )

        print(f"\nNet policy (top {args.top_k}):")
        print(f"  {'Move':<22} {'Net prob':>8}  {'bar'}")
        for m, prob in net_dist_sorted[: args.top_k]:
            bar = "#" * int(prob * 40)
            print(f"  {_move_label(m):<22} {prob:>8.4f}  {bar}")

        # KL divergence
        ab_map = {(m.source, m.tile, m.destination): p for m, p in ab_dist}
        net_map = {(m.source, m.tile, m.destination): p for m, p in net_dist}
        epsilon = 1e-8
        kl = 0.0
        for key, p_ab in ab_map.items():
            if p_ab <= 0:
                continue
            p_net = net_map.get(key, epsilon)
            kl += p_ab * math.log(p_ab / max(p_net, epsilon))
        kl_values.append(kl)
        print(f"\nKL divergence (AlphaBeta || Net): {kl:.4f}")
        print(f"  (0=perfect match, log(n_moves)={math.log(len(legal)):.2f}=random)")

        # ── MCTS probe ─────────────────────────────────────────────────────
        # Run a real search from this position and show visit counts.
        # This reveals what the search actually chooses, not just the raw
        # policy and value outputs. If MCTS picks floor moves despite the
        # policy head preferring pattern moves, the value head is at fault.
        print(f"\nMCTS probe ({args.mcts_sims} sims):")
        from neural.search_tree import (
            SearchTree,
            make_batch_policy_value_fn,
            make_policy_value_fn,
        )

        probe_tree = SearchTree(
            policy_value_fn=make_policy_value_fn(net),
            batch_policy_value_fn=make_batch_policy_value_fn(net),
            simulations=args.mcts_sims,
            temperature=0.0,
            batch_size=args.mcts_sims,
        )
        probe_tree.reset(game.clone())
        mcts_move = probe_tree.choose_move(game.clone())
        root = probe_tree._root
        if root and root.children:
            visited = [c for c in root.children if c.visits > 0]
            visited_sorted = sorted(visited, key=lambda c: c.visits, reverse=True)
            total_visits = sum(c.visits for c in visited)
            print(
                f"  {'Move':<22} {'visits':>7} {'visit%':>7} {'q(root)':>8}  {'chosen'}"
            )
            for child in visited_sorted[: args.top_k]:
                visit_pct = child.visits / total_visits * 100
                chosen_marker = (
                    " <-- MCTS"
                    if (
                        child.move is not None
                        and child.move.source == mcts_move.source
                        and child.move.tile == mcts_move.tile
                        and child.move.destination == mcts_move.destination
                    )
                    else ""
                )
                dest_type = (
                    "FLOOR"
                    if (child.move is not None and child.move.destination == FLOOR)
                    else "     "
                )
                # q_value stored from child (opponent) perspective — negate for root
                print(
                    f"  {_move_label(child.move):<22} {child.visits:>7} "
                    f"{visit_pct:>6.1f}% {-child.q_value:>+8.4f}  "
                    f"{dest_type}{chosen_marker}"
                )
            floor_children = [
                c for c in visited if c.move is not None and c.move.destination == FLOOR
            ]
            pattern_children = [
                c for c in visited if c.move is not None and c.move.destination != FLOOR
            ]
            if floor_children and pattern_children:
                # q_value is stored from child's perspective (opponent).
                # Negate to get root player's perspective.
                avg_floor_q = sum(-c.q_value for c in floor_children) / len(
                    floor_children
                )
                avg_pattern_q = sum(-c.q_value for c in pattern_children) / len(
                    pattern_children
                )
                print(
                    f"\n  avg q_value (root perspective) -- "
                    f"floor: {avg_floor_q:+.4f}  pattern: {avg_pattern_q:+.4f}"
                )
                if avg_floor_q > avg_pattern_q:
                    print("  *** VALUE HEAD PREFERS FLOOR MOVES ***")
                else:
                    print("  Value head correctly prefers pattern moves")

            # ── Top child subtree probe ────────────────────────────────────
            # Show what the net thinks of the position AFTER the most-visited
            # move. The net evaluates from the opponent's perspective there.
            # PUCT negates that value, so a negative child value = good for root.
            top_child = visited_sorted[0]
            if top_child.move is not None:
                child_game = top_child.game
                child_legal = child_game.legal_moves()
                _, cpw, cpd, cpa = _net_forward(
                    net, child_game, child_legal if child_legal else legal
                )
                print(
                    f"\n  Top child: {_move_label(top_child.move)} "
                    f"({top_child.visits} visits)"
                )
                print(
                    f"  Net value at child (opponent's view): "
                    f"v_win={cpw:+.4f}  v_diff={cpd:+.4f}  v_abs={cpa:+.4f}"
                )
                print(
                    f"  PUCT negates -> root player sees:     "
                    f"v_win={-cpw:+.4f}  v_diff={-cpd:+.4f}  v_abs={-cpa:+.4f}"
                )
                if top_child.move.destination == FLOOR:
                    if -cpd > 0:
                        print(
                            f"  *** Floor child looks good to root ({-cpd:+.4f}) "
                            f"— value head not penalizing floor ***"
                        )
                    else:
                        print(
                            f"  Floor child looks bad to root ({-cpd:+.4f}) "
                            f"— but still got most visits (prior effect?)"
                        )

        # Play the move
        game.make_move(move)
        game.advance()
        move_count += 1

    print("\n" + "=" * 70)
    if kl_values:
        avg_kl = sum(kl_values) / len(kl_values)
        print(f"Summary over {len(kl_values)} moves:")
        print(f"  avg KL divergence: {avg_kl:.4f}")
        print(f"  min KL:            {min(kl_values):.4f}")
        print(f"  max KL:            {max(kl_values):.4f}")
        print(f"  log(avg moves):    ~{math.log(70):.2f} (random baseline)")


if __name__ == "__main__":
    main()


from engine.game import Game
from neural.model import AzulNet
from neural.encoder import encode_state
import torch

net = AzulNet()
net.load_state_dict(torch.load("checkpoints/latest.pt", map_location="cpu"))
net.eval()

# Run 5 fresh games, encode state, get value_diff
diffs = []
for _ in range(5):
    g = Game()
    g.setup_round()
    spatial, flat = encode_state(g)
    with torch.no_grad():
        _, _, vd, _ = net(spatial.unsqueeze(0), flat.unsqueeze(0))
    diffs.append(vd.item())
print(diffs)
