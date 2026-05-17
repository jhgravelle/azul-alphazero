#!/usr/bin/env python3
"""Load and display the full game state encoding from a recorded turn."""

import json
import sys
from pathlib import Path
from typing import Any

from engine.constants import COLOR_TILES, Tile
from engine.game import Game
from engine.game_recorder import GameRecord
from engine.replay import replay_to_move


def color_char(color: Tile) -> str:
    """Convert a Tile enum to its single-character representation."""
    char_map = {
        Tile.BLUE: "B",
        Tile.YELLOW: "Y",
        Tile.RED: "R",
        Tile.BLACK: "K",
        Tile.WHITE: "W",
    }
    return char_map.get(color, "?")


def display_game_encoding(game: Game) -> None:
    """Display the full game state encoding with labeled sections and axes."""
    enc = game.encoded_features
    slices = Game.ENCODING_SLICES

    print("=" * 80)
    print(
        f"GAME STATE ENCODING - Round {game.round}, "
        f"Turn {game.turn} [Seed: {game.seed}]"
    )
    print("=" * 80)
    print()

    # ─── Round and Game Status ────────────────────────────────────────────
    print("ROUND AND GAME STATUS")
    print("-" * 80)
    s = slices["round"]
    print(f"  Round: {enc[s][0]}")
    s = slices["can_current_player_trigger"]
    print(f"  Can current player trigger: {enc[s][0]}")
    s = slices["can_opponent_trigger"]
    print(f"  Can opponent trigger: {enc[s][0]}")
    s = slices["has_game_end_been_triggered"]
    print(f"  Has game end been triggered: {enc[s][0]}")
    print()

    # ─── Tile Availability ────────────────────────────────────────────────
    print("TILE AVAILABILITY")
    print("-" * 80)
    print("  (How many of each color are available in factories/center)")
    s = slices["tile_availability"]
    avail_vals = enc[s]
    avail_str = "  ".join(
        f"{color_char(COLOR_TILES[i])}: {avail_vals[i]:2d}" for i in range(5)
    )
    print(f"    {avail_str}")
    print()

    # ─── Tile Source Count ────────────────────────────────────────────────
    print("TILE SOURCE COUNT")
    print("-" * 80)
    print("  (Number of sources where each color can be obtained)")
    s = slices["tile_source_count"]
    source_vals = enc[s]
    source_str = "  ".join(
        f"{color_char(COLOR_TILES[i])}: {source_vals[i]:2d}" for i in range(5)
    )
    print(f"    {source_str}")
    print()

    # ─── Bag State ────────────────────────────────────────────────────────
    print("BAG STATE")
    print("-" * 80)
    print("  (Number of tiles of each color remaining in bag)")
    s = slices["bag_state"]
    bag_vals = enc[s]
    bag_str = "  ".join(
        f"{color_char(COLOR_TILES[i])}: {bag_vals[i]:2d}" for i in range(5)
    )
    print(f"    {bag_str}")
    print()

    # ─── Current Player State ─────────────────────────────────────────────
    current_idx = game.current_player_index
    player_name = game.players[current_idx].name
    print(f"CURRENT PLAYER STATE (Player {current_idx + 1}: {player_name})")
    print("-" * 80)
    s = slices["current_player_encoded"]
    display_player_encoding(enc[s], game.players[current_idx])
    print()

    # ─── Opponent State ───────────────────────────────────────────────────
    opponent_idx = 1 - current_idx
    print(
        f"OPPONENT STATE (Player {opponent_idx + 1}: {game.players[opponent_idx].name})"
    )
    print("-" * 80)
    s = slices["opponent_encoded"]
    display_player_encoding(enc[s], game.players[opponent_idx])
    print()


def display_player_encoding(player_enc: list[int], player: Any) -> None:
    """Display a player's encoded state with labeled sections."""
    from engine.player import ENCODING_SLICES as PLAYER_SLICES

    # ─── Wall ─────────────────────────────────────────────────────────────
    print("  WALL (5x5 grid)")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["wall"]
    wall_vals = player_enc[s]
    print("  COL:     B    Y    R    K    W")
    for row in range(5):
        row_vals = wall_vals[row * 5 : (row + 1) * 5]
        row_label = f"  ROW {row + 1}: "
        row_str = "  ".join(f"{v:3d}" for v in row_vals)
        print(row_label + row_str)
    print()

    # ─── Pending Wall ──────────────────────────────────────────────────────
    print("  PENDING WALL (5x5 grid)")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["pending_wall"]
    pending_vals = player_enc[s]
    print("  COL:     B    Y    R    K    W")
    for row in range(5):
        row_vals = pending_vals[row * 5 : (row + 1) * 5]
        row_label = f"  ROW {row + 1}: "
        row_str = "  ".join(f"{v:3d}" for v in row_vals)
        print(row_label + row_str)
    print()

    # ─── Adjacency Grid ────────────────────────────────────────────────────
    print("  ADJACENCY GRID (5x5)")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["adjacency_grid"]
    adj_vals = player_enc[s]
    print("  COL:     B    Y    R    K    W")
    for row in range(5):
        row_vals = adj_vals[row * 5 : (row + 1) * 5]
        row_label = f"  ROW {row + 1}: "
        row_str = "  ".join(f"{v:3d}" for v in row_vals)
        print(row_label + row_str)
    print()

    # ─── Wall Row Demand ───────────────────────────────────────────────────
    print("  WALL ROW DEMAND")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["wall_row_demand"]
    row_demand_vals = player_enc[s]
    print("  COL:     B    Y    R    K    W")
    for row in range(5):
        row_vals = row_demand_vals[row * 5 : (row + 1) * 5]
        row_label = f"  ROW {row + 1}: "
        row_str = "  ".join(f"{v:3d}" for v in row_vals)
        print(row_label + row_str)
    print()

    # ─── Wall Column Demand ────────────────────────────────────────────────
    print("  WALL COLUMN DEMAND")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["wall_col_demand"]
    col_demand_vals = player_enc[s]
    print("  ROW:     B    Y    R    K    W")
    for col in range(5):
        col_vals = col_demand_vals[col * 5 : (col + 1) * 5]
        col_label = f"  COL {col + 1}: "
        col_str = "  ".join(f"{v:3d}" for v in col_vals)
        print(col_label + col_str)
    print()

    # ─── Wall Tile Demand ──────────────────────────────────────────────────
    print("  WALL TILE DEMAND")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["wall_tile_demand"]
    tile_demand_vals = player_enc[s]
    tile_str = "  ".join(
        f"{color_char(COLOR_TILES[i])}: {tile_demand_vals[i]:2d}" for i in range(5)
    )
    print(f"    {tile_str}")
    print()

    # ─── Pattern Line Demand ───────────────────────────────────────────────
    print("  PATTERN LINE DEMAND")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["pattern_demand"]
    pattern_demand_vals = player_enc[s]
    demand_str = "  ".join(
        f"{color_char(COLOR_TILES[i])}: {pattern_demand_vals[i]:2d}" for i in range(5)
    )
    print(f"    {demand_str}")
    print()

    # ─── Pattern Line Capacity ─────────────────────────────────────────────
    print("  PATTERN LINE CAPACITY")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["pattern_capacity"]
    capacity_vals = player_enc[s]
    print("  COL:     B    Y    R    K    W")
    for row in range(5):
        row_vals = capacity_vals[row * 5 : (row + 1) * 5]
        row_label = f"  ROW {row + 1}: "
        row_str = "  ".join(f"{v:3d}" for v in row_vals)
        print(row_label + row_str)
    print()

    # ─── Scoring & Misc ────────────────────────────────────────────────────
    print("  SCORING & MISC")
    print("  " + "-" * 76)
    s = PLAYER_SLICES["scoring"]
    scoring_vals = player_enc[s]
    s_misc = PLAYER_SLICES["misc"]
    misc_vals = player_enc[s_misc]
    print(f"    Official score:        {scoring_vals[0]}")
    print(f"    Pending points:        {scoring_vals[1]}")
    print(f"    Floor penalty:         {scoring_vals[2]}")
    print(f"    End-of-game bonus:     {scoring_vals[3]}")
    print(f"    Earned (projected):    {scoring_vals[4]}")
    print(f"    Has first player tile: {misc_vals[0]}")
    print(f"    Total tiles used:      {misc_vals[1]}")
    print(f"    Max pattern capacity:  {misc_vals[2]}")
    print()


def main() -> None:
    """Load a recording and display a specific turn's encoding."""
    if len(sys.argv) < 2:
        print(
            "Usage: python -m scripts.show_turn <recording_file> [turn_number] [--game]"
        )
        print()
        print("Arguments:")
        print("  recording_file   Path to a .json game recording")
        print("  turn_number      Which turn to display (default: 0, initial state)")
        print("  --game           Also display the original game state display")
        print()
        print("Examples:")
        print("  python -m scripts.show_turn recordings/game.json")
        print("      # Initial encoding")
        print("  python -m scripts.show_turn recordings/game.json 5")
        print("      # Turn 5 encoding")
        print("  python -m scripts.show_turn recordings/game.json 5 --game")
        print("      # With game display")
        sys.exit(1)

    recording_path = Path(sys.argv[1])
    if not recording_path.exists():
        print(f"Error: Recording file not found: {recording_path}")
        sys.exit(1)

    turn_number = 0
    show_game_state = False

    for i in range(2, len(sys.argv)):
        if sys.argv[i] == "--game":
            show_game_state = True
        else:
            try:
                turn_number = int(sys.argv[i])
            except ValueError:
                print(f"Error: Invalid argument: {sys.argv[i]}")
                sys.exit(1)

    # Load the recording
    try:
        record = GameRecord.load(recording_path)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading recording: {e}")
        sys.exit(1)

    # Reconstruct the game to the specified turn
    try:
        game = replay_to_move(record, turn_number)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Display game state if requested
    if show_game_state:
        print(game)
        print()
        print()

    # Display the encoding
    display_game_encoding(game)


if __name__ == "__main__":
    main()
