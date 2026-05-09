"""Tests for neural/encoder.py (v3 - flat MLP encoding).

Run with:
    pytest tests/test_encoder.py -v
"""

import pytest
import torch

from neural.encoder import (
    FLAT_SIZE,
    MOVE_SPACE_SIZE,
    OFF_BAG_COUNT,
    OFF_MY_FLOOR,
    OFF_MY_FP_TOKEN,
    OFF_MY_PATTERN,
    OFF_MY_SCORE,
    OFF_MY_WALL,
    OFF_OPP_FLOOR,
    OFF_OPP_FP_TOKEN,
    OFF_OPP_SCORE,
    OFF_OPP_WALL,
    OFF_SOURCES_WITH_COLOR,
    OFF_TILES_AVAILABLE,
    decode_move,
    encode_move,
    encode_state,
    format_encoding,
)
from engine.game import CENTER, FLOOR, Game, Move
from engine.constants import BOARD_SIZE, COLOR_TILES, COLUMN_FOR_TILE_IN_ROW, Tile


def fresh_game() -> Game:
    game = Game()
    game.setup_round()
    return game


# ── Shape ─────────────────────────────────────────────────────────────────


def test_encode_state_returns_correct_shape():
    game = fresh_game()
    encoding = encode_state(game)
    assert encoding.shape == (FLAT_SIZE,)


def test_encoding_size_is_125():
    assert FLAT_SIZE == 125


# ── Wall encoding ──────────────────────────────────────────────────────────


def test_wall_empty_at_game_start():
    game = fresh_game()
    encoding = encode_state(game)
    assert encoding[OFF_MY_WALL : OFF_MY_WALL + 25].sum().item() == 0.0
    assert encoding[OFF_OPP_WALL : OFF_OPP_WALL + 25].sum().item() == 0.0


def test_my_wall_reflects_placed_tile():
    game = fresh_game()
    blue = Tile.BLUE
    wall_col = COLUMN_FOR_TILE_IN_ROW[blue][0]
    game.current_player.wall[0][wall_col] = 1

    encoding = encode_state(game)
    flat_idx = 0 * BOARD_SIZE + wall_col
    assert encoding[OFF_MY_WALL + flat_idx].item() == 1.0
    assert encoding[OFF_MY_WALL : OFF_MY_WALL + 25].sum().item() == 1.0


def test_opponent_wall_reflects_opponent_tiles():
    game = fresh_game()
    opponent_index = 1 - game.current_player_index
    yellow = Tile.YELLOW
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][2]
    game.players[opponent_index].wall[2][wall_col] = 1

    encoding = encode_state(game)
    flat_idx = 2 * BOARD_SIZE + wall_col
    assert encoding[OFF_OPP_WALL + flat_idx].item() == 1.0
    assert encoding[OFF_MY_WALL : OFF_MY_WALL + 25].sum().item() == 0.0


# ── Pattern line fill ratio ────────────────────────────────────────────────


def test_pattern_line_empty_at_game_start():
    game = fresh_game()
    encoding = encode_state(game)
    assert encoding[OFF_MY_PATTERN : OFF_MY_PATTERN + 25].sum().item() == 0.0


def test_pattern_line_partial_fill():
    game = fresh_game()
    red = Tile.RED
    game.current_player.place(2, [red, red])
    wall_col = COLUMN_FOR_TILE_IN_ROW[red][2]

    encoding = encode_state(game)
    flat_idx = 2 * BOARD_SIZE + wall_col
    assert encoding[OFF_MY_PATTERN + flat_idx].item() == pytest.approx(2 / 3)


def test_pattern_line_full_ratio():
    game = fresh_game()
    yellow = Tile.YELLOW
    game.current_player.place(1, [yellow, yellow])
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][1]

    encoding = encode_state(game)
    flat_idx = 1 * BOARD_SIZE + wall_col
    assert encoding[OFF_MY_PATTERN + flat_idx].item() == pytest.approx(1.0)


def test_pattern_line_suppressed_when_wall_filled():
    game = fresh_game()
    yellow = Tile.YELLOW
    game.current_player.place(1, [yellow, yellow])
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][1]
    game.current_player.wall[1][wall_col] = 1

    encoding = encode_state(game)
    flat_idx = 1 * BOARD_SIZE + wall_col
    assert encoding[OFF_MY_PATTERN + flat_idx].item() == 0.0


# ── Scores and penalties ───────────────────────────────────────────────────


def test_scores_zero_at_game_start():
    game = fresh_game()
    encoding = encode_state(game)
    assert encoding[OFF_MY_SCORE].item() == 0.0
    assert encoding[OFF_OPP_SCORE].item() == 0.0


def test_official_score_normalized():
    game = fresh_game()
    game.current_player.score = 50
    encoding = encode_state(game)
    assert encoding[OFF_MY_SCORE].item() == pytest.approx(50 / 100.0)


def test_floor_penalty_normalized():
    game = fresh_game()
    game.current_player.floor_line.extend([Tile.BLUE, Tile.RED])
    game.current_player._update_penalty()

    encoding = encode_state(game)
    assert encoding[OFF_MY_FLOOR].item() != 0.0
    assert encoding[OFF_OPP_FLOOR].item() == 0.0


def test_first_player_token_on_my_floor():
    game = fresh_game()
    game.current_player.floor_line.append(Tile.FIRST_PLAYER)

    encoding = encode_state(game)
    assert encoding[OFF_MY_FP_TOKEN].item() == 1.0
    assert encoding[OFF_OPP_FP_TOKEN].item() == 0.0


def test_first_player_token_on_opponent_floor():
    game = fresh_game()
    opponent_index = 1 - game.current_player_index
    game.players[opponent_index].floor_line.append(Tile.FIRST_PLAYER)

    encoding = encode_state(game)
    assert encoding[OFF_MY_FP_TOKEN].item() == 0.0
    assert encoding[OFF_OPP_FP_TOKEN].item() == 1.0


# ── Game tiles and resources ──────────────────────────────────────────────


def test_tiles_available_zero_when_sources_empty():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    encoding = encode_state(game)
    tiles_slice = encoding[OFF_TILES_AVAILABLE : OFF_TILES_AVAILABLE + len(COLOR_TILES)]
    assert tiles_slice.sum().item() == 0.0


def test_tiles_available_counts_all_sources():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    game.factories[0].extend([Tile.BLUE] * 3)
    game.factories[1].extend([Tile.BLUE] * 2)
    game.center.extend([Tile.BLUE] * 1)
    blue_idx = COLOR_TILES.index(Tile.BLUE)

    encoding = encode_state(game)
    assert encoding[OFF_TILES_AVAILABLE + blue_idx].item() == pytest.approx(6 / 20)


def test_sources_with_color_zero_when_sources_empty():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    encoding = encode_state(game)
    sources_slice = encoding[
        OFF_SOURCES_WITH_COLOR : OFF_SOURCES_WITH_COLOR + len(COLOR_TILES)
    ]
    assert sources_slice.sum().item() == 0.0


def test_sources_with_color_counts_distinct_sources():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    game.factories[0].extend([Tile.RED] * 4)
    game.factories[1].extend([Tile.RED] * 2)
    red_idx = COLOR_TILES.index(Tile.RED)

    encoding = encode_state(game)
    assert encoding[OFF_SOURCES_WITH_COLOR + red_idx].item() == pytest.approx(2 / 5)


def test_bag_count_zero_when_bag_empty():
    game = fresh_game()
    game.bag.clear()

    encoding = encode_state(game)
    bag_slice = encoding[OFF_BAG_COUNT : OFF_BAG_COUNT + len(COLOR_TILES)]
    assert bag_slice.sum().item() == 0.0


def test_bag_count_normalized_correctly():
    game = fresh_game()
    game.bag.clear()
    game.bag.extend([Tile.BLUE] * 10)
    blue_idx = COLOR_TILES.index(Tile.BLUE)

    encoding = encode_state(game)
    assert encoding[OFF_BAG_COUNT + blue_idx].item() == pytest.approx(10 / 20)


def test_encoding_all_finite():
    game = fresh_game()
    encoding = encode_state(game)
    assert torch.isfinite(encoding).all()


# ── Move encoding ──────────────────────────────────────────────────────────


def test_move_encode_decode_roundtrip_factory_to_line():
    game = fresh_game()
    move = game.legal_moves()[0]
    index = encode_move(move, game)
    recovered = decode_move(index, game)
    assert recovered.source == move.source
    assert recovered.tile == move.tile
    assert recovered.destination == move.destination


def test_move_encode_decode_roundtrip_center_to_floor():
    game = fresh_game()
    move = Move(source=CENTER, tile=Tile.BLUE, destination=FLOOR)
    index = encode_move(move, game)
    recovered = decode_move(index, game)
    assert recovered.source == CENTER
    assert recovered.tile == Tile.BLUE
    assert recovered.destination == FLOOR


def test_move_index_in_valid_range():
    game = fresh_game()
    for move in game.legal_moves():
        index = encode_move(move, game)
        assert 0 <= index < MOVE_SPACE_SIZE


def test_all_legal_moves_have_distinct_indices():
    game = fresh_game()
    moves = game.legal_moves()
    indices = [encode_move(m, game) for m in moves]
    assert len(indices) == len(set(indices))


def test_format_encoding_five_grids_side_by_side():
    game = fresh_game()
    encoding = encode_state(game)
    result = format_encoding(encoding)
    lines = result.splitlines()
    assert len(lines) == 5
    assert all(len(line) == len(lines[0]) for line in lines)
