# cli/cli.py

"""Terminal interface for Azul — human vs human, full game state displayed each turn.

Run from the project root:
    python -m cli.cli
"""

import os
import re
from engine.game import FLOOR, WALL_PATTERN, Game, Move
from engine.tile import Tile
from engine.constants import SIZE

# Enable ANSI escape codes on Windows
os.system("color")

# ── ANSI color codes ───────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
CYAN = "\033[96m"
WHITE = "\033[97m"

# ── Tile rendering ─────────────────────────────────────────────────────────────

_TILE_STYLE: dict[Tile, tuple[str, str]] = {
    Tile.BLUE: (BLUE, "B"),
    Tile.YELLOW: (YELLOW, "Y"),
    Tile.RED: (RED, "R"),
    Tile.BLACK: (GRAY, "K"),
    Tile.WHITE: (CYAN, "W"),
    Tile.FIRST_PLAYER: (WHITE, "1"),
}

# Accepts both number and letter for color input
_COLOR_INPUT: dict[str, Tile] = {
    "1": Tile.BLUE,
    "B": Tile.BLUE,
    "2": Tile.YELLOW,
    "Y": Tile.YELLOW,
    "3": Tile.RED,
    "R": Tile.RED,
    "4": Tile.BLACK,
    "K": Tile.BLACK,
    "5": Tile.WHITE,
    "W": Tile.WHITE,
}

_SOURCE_INPUT: dict[str, int] = {str(i): i - 1 for i in range(1, 6)}
_SOURCE_INPUT["0"] = -1  # CENTER


def render_tile(tile: Tile | None, hint: Tile | None = None) -> str:
    """Return a colored single-character string for a tile.

    If tile is None and hint is provided, renders a dim dot in the hint color
    (used for empty wall cells to show which color belongs there).
    """
    if tile is None:
        if hint is not None:
            color, _ = _TILE_STYLE[hint]
            return f"{DIM}{color}.{RESET}"
        return f"{DIM}.{RESET}"
    color, symbol = _TILE_STYLE[tile]
    return f"{color}{symbol}{RESET}"


# ── Board rendering ────────────────────────────────────────────────────────────


def visible_len(s: str) -> int:
    """Return the visible length of a string, ignoring ANSI escape codes."""
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def render_player_board(player, player_idx: int) -> list[str]:
    """Return a list of lines representing one player's board.

    Layout per row:
        pattern line (right-aligned, capacity-wide) | row number | wall row
    """
    lines = []
    lines.append(f"{BOLD}Player {player_idx + 1}  Score: {player.score}{RESET}")
    lines.append("")
    lines.append("  Pattern lines    Wall")

    for row in range(SIZE):
        cap = row + 1
        pattern = player.pattern_lines[row]

        # Pattern line — only cap dots wide, right-aligned in BOARD_SIZE field
        pattern_cells = [render_tile(None)] * (cap - len(pattern))
        pattern_cells += [render_tile(t) for t in pattern]
        padding = " " * (SIZE - cap) * 2
        pattern_str = padding + " ".join(pattern_cells)

        # Wall row with dim colored hints for empty cells
        wall_cells = [
            render_tile(player.wall[row][col], hint=WALL_PATTERN[row][col])
            for col in range(SIZE)
        ]
        wall_str = " ".join(wall_cells)

        lines.append(f"  {pattern_str}  {row + 1}  {wall_str}")

    # Floor line
    if player.floor_line:
        floor_str = " ".join(render_tile(t) for t in player.floor_line)
    else:
        floor_str = f"{DIM}empty{RESET}"
    lines.append("")
    lines.append(f"  Floor: {floor_str}")

    return lines


def render_both_boards(game: Game) -> str:
    """Render both player boards side by side."""
    boards = [render_player_board(p, i) for i, p in enumerate(game.state.players)]

    max_lines = max(len(b) for b in boards)
    for b in boards:
        while len(b) < max_lines:
            b.append("")

    col_width = max(visible_len(line) for line in boards[0]) + 4
    result = []
    for left, right in zip(boards[0], boards[1]):
        padding = " " * (col_width - visible_len(left))
        result.append(left + padding + right)
    return "\n".join(result)


def render_factories(game: Game) -> str:
    """Render the factory displays and center pool."""
    lines = [f"{BOLD}Factories:{RESET}"]
    for i, factory in enumerate(game.state.factories):
        if factory:
            tiles_str = " ".join(render_tile(t) for t in factory)
            lines.append(f"  [{i + 1}] {tiles_str}")
        else:
            lines.append(f"  [{i + 1}] {DIM}empty{RESET}")

    center = game.state.center
    if center:
        center_str = " ".join(render_tile(t) for t in center)
        lines.append(f"  [0] center: {center_str}")
    else:
        lines.append(f"  [0] center: {DIM}empty{RESET}")

    return "\n".join(lines)


def render_color_key() -> str:
    """Render a one-line color reference."""
    parts = [f"{BOLD}Colors:{RESET}"]
    for n, (ch, tile) in enumerate(
        [
            ("B", Tile.BLUE),
            ("Y", Tile.YELLOW),
            ("R", Tile.RED),
            ("K", Tile.BLACK),
            ("W", Tile.WHITE),
        ],
        start=1,
    ):
        color, symbol = _TILE_STYLE[tile]
        parts.append(f"  {n}/{ch}={color}{symbol}{RESET}")
    return "".join(parts)


def render_input_hint() -> str:
    """Render the input legend shown below the factories."""
    return (
        f"  {BOLD}Source:{RESET} 0=center, 1–5=factory  "
        f"{BOLD}Destination:{RESET} 0=floor, 1–5=row"
    )


# ── Input handling ─────────────────────────────────────────────────────────────


def parse_input(prompt: str, valid: set[str]) -> str:
    """Prompt until the user enters one of the valid characters."""
    while True:
        raw = input(prompt).strip().upper()
        if raw in valid:
            return raw
        print(f"  Please enter one of: {', '.join(sorted(valid))}")


def get_move_choice(moves: list[Move], current_player: int) -> Move:
    """Prompt the current player with three separate inputs. Loops until legal."""
    player_label = f"{BOLD}Player {current_player + 1}{RESET}"
    color_valid = set(_COLOR_INPUT.keys())
    while True:
        print(f"\n  {player_label}'s move:")
        color_ch = parse_input("  Color      (1/B 2/Y 3/R 4/K 5/W): ", color_valid)
        src_ch = parse_input("  Source     (0=center, 1–5=factory): ", set("012345"))
        dst_ch = parse_input("  Destination(0=floor,  1–5=row):     ", set("012345"))

        color = _COLOR_INPUT[color_ch]
        source = _SOURCE_INPUT[src_ch]
        destination = FLOOR if dst_ch == "0" else int(dst_ch) - 1

        move = Move(source=source, tile=color, destination=destination)
        if move in moves:
            return move
        print("  That's not a legal move — try again.")


# ── Game loop ──────────────────────────────────────────────────────────────────


def render_separator() -> str:
    return "═" * 60


def play() -> None:
    """Main game loop."""
    game = Game()
    round_num = 1

    while True:
        game.setup_round()

        while any(game.state.factories) or game.state.center:
            os.system("cls")

            current = game.state.current_player
            print(render_separator())
            print(f"{BOLD}Round {round_num} — Player {current + 1}'s turn{RESET}")
            print(render_separator())
            print()
            print(render_color_key())
            print()
            print(render_both_boards(game))
            print()
            print(render_factories(game))
            print()
            print(render_input_hint())

            move = get_move_choice(game.legal_moves(), current)
            game.make_move(move)

        game.score_round()
        round_num += 1

        if game.is_game_over():
            game.score_game()
            os.system("cls")
            print(render_separator())
            print(f"{BOLD}Game over!{RESET}")
            print(render_separator())
            print()
            print(render_both_boards(game))
            print()

            scores = [(p.score, i) for i, p in enumerate(game.state.players)]
            scores.sort(reverse=True)
            if scores[0][0] == scores[1][0]:
                print(f"{BOLD}It's a tie! Both players scored {scores[0][0]}.{RESET}")
            else:
                winner = scores[0][1] + 1
                print(
                    f"{BOLD}Player {winner} wins " f"with {scores[0][0]} points!{RESET}"
                )
            break


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        pass
