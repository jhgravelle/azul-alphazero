"""Tests for neural/encoder.py (v2).

Run with:
    pytest tests/test_encoder.py -v
"""

import pytest
import torch

from neural.encoder import (
    FLAT_SIZE,
    MOVE_SPACE_SIZE,
    NUM_CHANNELS,
    SPATIAL_SHAPE,
    CH_MY_WALL,
    CH_MY_PATTERN,
    CH_MY_BONUS,
    CH_OPP_WALL,
    CH_BAG,
    CH_SOURCE_DIST,
    OFF_MY_SCORE,
    OFF_OPP_SCORE,
    OFF_MY_FLOOR,
    OFF_OPP_FLOOR,
    OFF_MY_FP_TOKEN,
    OFF_OPP_FP_TOKEN,
    encode_state,
    encode_move,
    decode_move,
)
from engine.game import Game, CENTER, FLOOR
from engine.constants import (
    BOARD_SIZE,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
    WALL_PATTERN,
    Tile,
)


def fresh_game() -> Game:
    game = Game()
    game.setup_round()
    return game


def test_encode_state_returns_correct_shapes():
    game = fresh_game()
    spatial, flat = encode_state(game)
    assert spatial.shape == SPATIAL_SHAPE
    assert flat.shape == (FLAT_SIZE,)


def test_spatial_shape_constants():
    assert SPATIAL_SHAPE == (8, 5, 5)
    assert FLAT_SIZE == 8
    assert NUM_CHANNELS == 8


def test_wall_filled_channel_empty_at_game_start():
    game = fresh_game()
    spatial, _ = encode_state(game)
    assert spatial[CH_MY_WALL].sum().item() == 0.0
    assert spatial[CH_OPP_WALL].sum().item() == 0.0


def test_wall_filled_channel_reflects_placed_tile():
    game = fresh_game()
    blue = Tile.BLUE
    wall_col = COLUMN_FOR_TILE_IN_ROW[blue][0]
    game.current_player.wall[0][wall_col] = blue

    spatial, _ = encode_state(game)
    assert spatial[CH_MY_WALL, 0, wall_col].item() == 1.0
    assert spatial[CH_MY_WALL].sum().item() == 1.0


def test_wall_filled_opponent_channel_reflects_opponent_wall():
    game = fresh_game()
    opponent_index = 1 - game.current_player_index
    yellow = Tile.YELLOW
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][2]
    game.players[opponent_index].wall[2][wall_col] = yellow

    spatial, _ = encode_state(game)
    assert spatial[CH_OPP_WALL, 2, wall_col].item() == 1.0
    assert spatial[CH_MY_WALL].sum().item() == 0.0


def test_pattern_line_empty_produces_zero_channel():
    game = fresh_game()
    spatial, _ = encode_state(game)
    assert spatial[CH_MY_PATTERN].sum().item() == 0.0


def test_pattern_line_partial_fill_lands_at_correct_wall_col():
    game = fresh_game()
    red = Tile.RED
    game.current_player.pattern_lines[2] = [red, red]
    wall_col = COLUMN_FOR_TILE_IN_ROW[red][2]

    spatial, _ = encode_state(game)
    assert spatial[CH_MY_PATTERN, 2, wall_col].item() == pytest.approx(2 / 3)
    assert (spatial[CH_MY_PATTERN] != 0.0).sum().item() == 1


def test_pattern_line_full_still_shows_ratio_when_wall_empty():
    game = fresh_game()
    yellow = Tile.YELLOW
    game.current_player.pattern_lines[1] = [yellow, yellow]
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][1]

    spatial, _ = encode_state(game)
    assert spatial[CH_MY_PATTERN, 1, wall_col].item() == pytest.approx(1.0)


def test_pattern_line_suppressed_when_wall_cell_already_filled():
    game = fresh_game()
    yellow = Tile.YELLOW
    game.current_player.pattern_lines[1] = [yellow, yellow]
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][1]
    game.current_player.wall[1][wall_col] = yellow

    spatial, _ = encode_state(game)
    assert spatial[CH_MY_PATTERN, 1, wall_col].item() == 0.0


def test_bonus_proximity_decreases_as_wall_fills():
    game = fresh_game()
    spatial_before, _ = encode_state(game)
    proximity_before = spatial_before[CH_MY_BONUS].sum().item()

    for wall_col in range(BOARD_SIZE):
        color = WALL_PATTERN[0][wall_col]
        game.current_player.wall[0][wall_col] = color

    spatial_after, _ = encode_state(game)
    proximity_after = spatial_after[CH_MY_BONUS].sum().item()
    assert proximity_after < proximity_before


def test_bonus_proximity_range():
    game = fresh_game()
    for row in range(BOARD_SIZE):
        for wall_col in range(BOARD_SIZE):
            game.current_player.wall[row][wall_col] = WALL_PATTERN[row][wall_col]

    spatial, _ = encode_state(game)
    assert spatial[CH_MY_BONUS].max().item() <= 1.0 + 1e-6

    game2 = fresh_game()
    spatial2, _ = encode_state(game2)
    assert torch.isfinite(spatial2[CH_MY_BONUS]).all()


def test_bag_count_broadcast_across_rows():
    game = fresh_game()
    spatial, _ = encode_state(game)
    for color_idx in range(BOARD_SIZE):
        col_values = spatial[CH_BAG, :, color_idx]
        assert col_values.min().item() == pytest.approx(col_values.max().item())


def test_bag_count_zero_when_bag_empty():
    game = fresh_game()
    game.bag.clear()
    spatial, _ = encode_state(game)
    assert spatial[CH_BAG].sum().item() == 0.0


def test_bag_count_normalized_correctly():
    game = fresh_game()
    game.bag.clear()
    game.bag.extend([Tile.BLUE] * 10)
    blue_col = COLOR_TILES.index(Tile.BLUE)

    spatial, _ = encode_state(game)
    for row in range(BOARD_SIZE):
        assert spatial[CH_BAG, row, blue_col].item() == pytest.approx(10 / 20)


def test_source_distribution_zero_when_no_tiles_available():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    spatial, _ = encode_state(game)
    assert spatial[CH_SOURCE_DIST].sum().item() == 0.0


def test_source_distribution_single_source_with_four_tiles():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    game.factories[0].extend([Tile.BLUE] * 4)
    blue_col = COLOR_TILES.index(Tile.BLUE)

    spatial, _ = encode_state(game)
    assert spatial[CH_SOURCE_DIST, 3, blue_col].item() == pytest.approx(1 / 5)
    for bucket in range(5):
        if bucket != 3:
            assert spatial[CH_SOURCE_DIST, bucket, blue_col].item() == 0.0


def test_source_distribution_bucket_4_catches_five_plus():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    game.center.extend([Tile.BLUE] * 5)
    blue_col = COLOR_TILES.index(Tile.BLUE)

    spatial, _ = encode_state(game)
    assert spatial[CH_SOURCE_DIST, 4, blue_col].item() == pytest.approx(1 / 5)


def test_source_distribution_two_sources_same_color():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    game.factories[0].extend([Tile.RED] * 2)
    game.factories[1].extend([Tile.RED] * 2)
    red_col = COLOR_TILES.index(Tile.RED)

    spatial, _ = encode_state(game)
    assert spatial[CH_SOURCE_DIST, 1, red_col].item() == pytest.approx(2 / 5)


def test_flat_scores_zero_at_game_start():
    game = fresh_game()
    _, flat = encode_state(game)
    assert flat[OFF_MY_SCORE].item() == 0.0
    assert flat[OFF_OPP_SCORE].item() == 0.0


def test_flat_floor_penalty_negative():
    game = fresh_game()
    game.current_player.floor_line.extend([Tile.BLUE, Tile.RED])
    game.current_player._update_penalty()

    _, flat = encode_state(game)
    assert flat[OFF_MY_FLOOR].item() == pytest.approx(-2 / 14)
    assert flat[OFF_OPP_FLOOR].item() == 0.0


def test_flat_first_player_token_on_my_floor():
    game = fresh_game()
    game.current_player.floor_line.append(Tile.FIRST_PLAYER)

    _, flat = encode_state(game)
    assert flat[OFF_MY_FP_TOKEN].item() == 1.0
    assert flat[OFF_OPP_FP_TOKEN].item() == 0.0


def test_flat_first_player_token_on_opponent_floor():
    game = fresh_game()
    opponent_index = 1 - game.current_player_index
    game.players[opponent_index].floor_line.append(Tile.FIRST_PLAYER)

    _, flat = encode_state(game)
    assert flat[OFF_MY_FP_TOKEN].item() == 0.0
    assert flat[OFF_OPP_FP_TOKEN].item() == 1.0


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
    from engine.game import Move

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
