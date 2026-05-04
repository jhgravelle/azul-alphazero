# neural/zobrist.py
"""Zobrist hashing for Azul game states within a round.

Only the features that change move-to-move are hashed:
    - Pattern lines (both players)
    - Floor lines (both players)
    - Factories
    - Center (per-color counts + first player token)
    - Current player

Wall, scores, bag, and discard are frozen within a round and excluded.

Usage
-----
    table = ZobristTable()
    h = table.hash_state(game)          # full hash from scratch
    # or incrementally — XOR out old, XOR in new (future optimisation)
"""

from __future__ import annotations

import random
from engine.game import Game
from engine.constants import Tile, COLOR_TILES, BOARD_SIZE, TILES_PER_COLOR, PLAYERS

# Maximum center count per color — TILES_PER_COLOR is a safe upper bound.
_MAX_CENTER_COUNT = TILES_PER_COLOR + 1  # indices 0 … TILES_PER_COLOR
_MAX_FLOOR = 8  # 0 … 7 tiles on floor
_MAX_FACTORY_COUNT = 5  # 0 … 4 tiles of one color per factory
_NUM_FACTORIES = 2 * PLAYERS + 1  # 5 factories


class ZobristTable:
    """Immutable table of random 64-bit integers for Zobrist hashing.

    One instance should be created at program startup and reused for the
    lifetime of the search.
    """

    def __init__(self, seed: int = 42) -> None:
        rng = random.Random(seed)

        def rand() -> int:
            return rng.getrandbits(64)

        num_colors = len(COLOR_TILES)

        # Pattern lines: player × row × color_idx × count
        # color_idx=num_colors means "empty" (no color)
        # count range: 0 … row+1 capacity, but we allocate BOARD_SIZE+1 for simplicity
        self._pattern: list[list[list[list[int]]]] = [
            [
                [
                    [rand() for _ in range(BOARD_SIZE + 1)]  # count 0…5
                    for _ in range(num_colors + 1)  # color + empty
                ]
                for _ in range(BOARD_SIZE)  # row 0…4
            ]
            for _ in range(PLAYERS)  # player 0, 1
        ]

        # Floor lines: player × count (0…7)
        self._floor: list[list[int]] = [
            [rand() for _ in range(_MAX_FLOOR)] for _ in range(PLAYERS)
        ]

        # Factories: factory_idx × color_idx × count (0…4)
        self._factory: list[list[list[int]]] = [
            [[rand() for _ in range(_MAX_FACTORY_COUNT)] for _ in range(num_colors)]
            for _ in range(_NUM_FACTORIES)
        ]

        # Center: color_idx × count (0…TILES_PER_COLOR)
        self._center: list[list[int]] = [
            [rand() for _ in range(_MAX_CENTER_COUNT)] for _ in range(num_colors)
        ]

        # Center first player token present
        self._center_fp: int = rand()

        # Current player (0 or 1)
        self._current_player: list[int] = [rand(), rand()]

    # ── Public API ─────────────────────────────────────────────────────────

    def hash_state(self, game: Game) -> int:
        """Compute the Zobrist hash of the current game state from scratch."""
        h = 0
        num_colors = len(COLOR_TILES)

        h ^= self._current_player[game.current_player_index]

        for player_idx in range(PLAYERS):
            player = game.players[player_idx]
            for row in range(BOARD_SIZE):
                line = player.pattern_lines[row]
                if line:
                    color_idx = COLOR_TILES.index(line[0])
                    count = len(line)
                else:
                    color_idx = num_colors
                    count = 0
                h ^= self._pattern[player_idx][row][color_idx][count]

        for player_idx in range(PLAYERS):
            player = game.players[player_idx]
            floor_count = min(len(player.floor_line), _MAX_FLOOR - 1)
            h ^= self._floor[player_idx][floor_count]

        for f_idx, factory in enumerate(game.factories):
            for color_idx, color in enumerate(COLOR_TILES):
                count = min(factory.count(color), _MAX_FACTORY_COUNT - 1)
                if count > 0:
                    h ^= self._factory[f_idx][color_idx][count]

        for color_idx, color in enumerate(COLOR_TILES):
            count = min(game.center.count(color), _MAX_CENTER_COUNT - 1)
            if count > 0:
                h ^= self._center[color_idx][count]

        if Tile.FIRST_PLAYER in game.center:
            h ^= self._center_fp

        return h
