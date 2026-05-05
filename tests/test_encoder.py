"""Tests for neural/encoder.py (v3).

Run with:
    pytest tests/test_encoder.py -v
"""

import pytest
import torch

from neural.encoder import (
    EARNED_DIVISOR,
    FLAT_SIZE,
    MOVE_SPACE_SIZE,
    NUM_CHANNELS,
    OFF_BAG_COUNT,
    OFF_MY_COL_COMPLETION,
    OFF_MY_COLOR_COMPLETION,
    OFF_MY_FLOOR,
    OFF_MY_FP_TOKEN,
    OFF_MY_ROW_COMPLETION,
    OFF_MY_SCORE,
    OFF_OPP_FLOOR,
    OFF_OPP_FP_TOKEN,
    OFF_OPP_ROW_COMPLETION,
    OFF_OPP_SCORE,
    OFF_SOURCES_WITH_COLOR,
    OFF_TILES_AVAILABLE,
    SPATIAL_SHAPE,
    CH_MY_PATTERN,
    CH_MY_WALL,
    CH_OPP_WALL,
    decode_move,
    encode_move,
    encode_state,
)
from engine.game import Game, CENTER, FLOOR, Move
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


# ── Shape constants ────────────────────────────────────────────────────────


def test_encode_state_returns_correct_shapes():
    game = fresh_game()
    spatial, flat = encode_state(game)
    assert spatial.shape == SPATIAL_SHAPE
    assert flat.shape == (FLAT_SIZE,)


def test_spatial_shape_constants():
    assert SPATIAL_SHAPE == (4, 5, 5)
    assert FLAT_SIZE == 53
    assert NUM_CHANNELS == 4


# ── Wall filled channel ────────────────────────────────────────────────────


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


# ── Pattern line channel ───────────────────────────────────────────────────


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


# ── Flat scores ────────────────────────────────────────────────────────────


def test_flat_scores_zero_at_game_start():
    game = fresh_game()
    _, flat = encode_state(game)
    assert flat[OFF_MY_SCORE].item() == 0.0
    assert flat[OFF_OPP_SCORE].item() == 0.0


def test_flat_earned_uses_divisor_100():
    game = fresh_game()
    game.current_player.score = 50
    _, flat = encode_state(game)
    assert flat[OFF_MY_SCORE].item() == pytest.approx(50 / 100)
    assert EARNED_DIVISOR == 100.0


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


# ── Flat completion stats ──────────────────────────────────────────────────


def test_row_completion_zero_at_game_start():
    game = fresh_game()
    _, flat = encode_state(game)
    row_slice = flat[OFF_MY_ROW_COMPLETION : OFF_MY_ROW_COMPLETION + BOARD_SIZE]
    assert row_slice.sum().item() == 0.0


def test_row_completion_full_row():
    game = fresh_game()
    for wall_col in range(BOARD_SIZE):
        color = WALL_PATTERN[0][wall_col]
        game.current_player.wall[0][wall_col] = color

    _, flat = encode_state(game)
    assert flat[OFF_MY_ROW_COMPLETION].item() == pytest.approx(1.0)


def test_row_completion_partial_row():
    game = fresh_game()
    # Fill 2 of 5 cells in row 2
    for wall_col in range(2):
        color = WALL_PATTERN[2][wall_col]
        game.current_player.wall[2][wall_col] = color

    _, flat = encode_state(game)
    # weighted fill = (2+1)*2 = 6; divisor = (2+1)*5 = 15; ratio = 6/15 = 0.4
    assert flat[OFF_MY_ROW_COMPLETION + 2].item() == pytest.approx(6 / 15)


def test_row_completion_uses_post_placement_wall():
    game = fresh_game()
    yellow = Tile.YELLOW
    game.current_player.pattern_lines[0] = [yellow]  # full (capacity 1)

    _, flat = encode_state(game)
    # Post-placement wall places yellow in row 0; row 0 completion should be nonzero
    assert flat[OFF_MY_ROW_COMPLETION].item() > 0.0


def test_col_completion_zero_at_game_start():
    game = fresh_game()
    _, flat = encode_state(game)
    col_slice = flat[OFF_MY_COL_COMPLETION : OFF_MY_COL_COMPLETION + BOARD_SIZE]
    assert col_slice.sum().item() == 0.0


def test_col_completion_full_col():
    game = fresh_game()
    wall_col = 0
    for row in range(BOARD_SIZE):
        color = WALL_PATTERN[row][wall_col]
        game.current_player.wall[row][wall_col] = color

    _, flat = encode_state(game)
    assert flat[OFF_MY_COL_COMPLETION].item() == pytest.approx(1.0)


def test_color_completion_zero_at_game_start():
    game = fresh_game()
    _, flat = encode_state(game)
    color_slice = flat[
        OFF_MY_COLOR_COMPLETION : OFF_MY_COLOR_COMPLETION + len(COLOR_TILES)
    ]
    assert color_slice.sum().item() == 0.0


def test_color_completion_full_color():
    game = fresh_game()
    blue = Tile.BLUE
    for row in range(BOARD_SIZE):
        wall_col = COLUMN_FOR_TILE_IN_ROW[blue][row]
        game.current_player.wall[row][wall_col] = blue

    _, flat = encode_state(game)
    blue_idx = COLOR_TILES.index(blue)
    assert flat[OFF_MY_COLOR_COMPLETION + blue_idx].item() == pytest.approx(1.0)


def test_opponent_completion_tracks_opponent_wall():
    game = fresh_game()
    opponent_index = 1 - game.current_player_index
    for wall_col in range(BOARD_SIZE):
        color = WALL_PATTERN[0][wall_col]
        game.players[opponent_index].wall[0][wall_col] = color

    _, flat = encode_state(game)
    assert flat[OFF_MY_ROW_COMPLETION].item() == 0.0
    assert flat[OFF_OPP_ROW_COMPLETION].item() == pytest.approx(1.0)


# ── Flat game tiles ────────────────────────────────────────────────────────


def test_tiles_available_zero_when_sources_empty():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    _, flat = encode_state(game)
    tiles_slice = flat[OFF_TILES_AVAILABLE : OFF_TILES_AVAILABLE + len(COLOR_TILES)]
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

    _, flat = encode_state(game)
    assert flat[OFF_TILES_AVAILABLE + blue_idx].item() == pytest.approx(6 / 20)


def test_sources_with_color_zero_when_sources_empty():
    game = fresh_game()
    for factory in game.factories:
        factory.clear()
    game.center.clear()

    _, flat = encode_state(game)
    sources_slice = flat[
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

    _, flat = encode_state(game)
    assert flat[OFF_SOURCES_WITH_COLOR + red_idx].item() == pytest.approx(2 / 5)


def test_bag_count_zero_when_bag_empty():
    game = fresh_game()
    game.bag.clear()

    _, flat = encode_state(game)
    bag_slice = flat[OFF_BAG_COUNT : OFF_BAG_COUNT + len(COLOR_TILES)]
    assert bag_slice.sum().item() == 0.0


def test_bag_count_normalized_correctly():
    game = fresh_game()
    game.bag.clear()
    game.bag.extend([Tile.BLUE] * 10)
    blue_idx = COLOR_TILES.index(Tile.BLUE)

    _, flat = encode_state(game)
    assert flat[OFF_BAG_COUNT + blue_idx].item() == pytest.approx(10 / 20)


def test_flat_all_finite():
    game = fresh_game()
    _, flat = encode_state(game)
    assert torch.isfinite(flat).all()


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
