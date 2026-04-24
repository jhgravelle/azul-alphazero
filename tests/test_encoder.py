"""Tests for neural/encoder.py (v2).

Run with:
    pytest tests/test_encoder.py -v
"""

import pytest

from neural.encoder import (
    FLAT_SIZE,
    MOVE_SPACE_SIZE,
    NUM_CHANNELS,
    SPATIAL_SHAPE,
    CH_MY_WALL,
    CH_MY_PATTERN,
    CH_MY_BONUS,
    CH_OPP_WALL,
    CH_OPP_BONUS,
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


# ── Fixtures ───────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    game = Game()
    game.setup_round()
    return game


# ── Shape tests ────────────────────────────────────────────────────────────


def test_encode_state_returns_correct_shapes():
    game = fresh_game()
    spatial, flat = encode_state(game)
    assert spatial.shape == SPATIAL_SHAPE
    assert flat.shape == (FLAT_SIZE,)


def test_spatial_shape_constants():
    assert SPATIAL_SHAPE == (8, 5, 5)
    assert FLAT_SIZE == 8
    assert NUM_CHANNELS == 8


# ── Wall filled channel ────────────────────────────────────────────────────


def test_wall_filled_channel_empty_at_game_start():
    game = fresh_game()
    spatial, _ = encode_state(game)
    assert spatial[CH_MY_WALL].sum().item() == 0.0
    assert spatial[CH_OPP_WALL].sum().item() == 0.0


def test_wall_filled_channel_reflects_placed_tile():
    game = fresh_game()
    # Place blue in row 0 — blue's wall column in row 0 is 0
    blue = Tile.BLUE
    wall_col = COLUMN_FOR_TILE_IN_ROW[blue][0]
    current_player = game.state.current_player
    game.state.players[current_player].wall[0][wall_col] = blue

    spatial, _ = encode_state(game)
    assert spatial[CH_MY_WALL, 0, wall_col].item() == 1.0
    # All other cells in my wall channel should be 0
    total = spatial[CH_MY_WALL].sum().item()
    assert total == 1.0


def test_wall_filled_opponent_channel_reflects_opponent_wall():
    game = fresh_game()
    opponent = 1 - game.state.current_player
    yellow = Tile.YELLOW
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][2]
    game.state.players[opponent].wall[2][wall_col] = yellow

    spatial, _ = encode_state(game)
    assert spatial[CH_OPP_WALL, 2, wall_col].item() == 1.0
    assert spatial[CH_MY_WALL].sum().item() == 0.0


# ── Pattern line fill ratio channel ───────────────────────────────────────


def test_pattern_line_empty_produces_zero_channel():
    game = fresh_game()
    spatial, _ = encode_state(game)
    assert spatial[CH_MY_PATTERN].sum().item() == 0.0


def test_pattern_line_partial_fill_lands_at_correct_wall_col():
    game = fresh_game()
    current_player = game.state.current_player
    board = game.state.players[current_player]

    # Fill row 2 (capacity 3) with 2 red tiles
    red = Tile.RED
    board.pattern_lines[2] = [red, red]
    wall_col = COLUMN_FOR_TILE_IN_ROW[red][2]

    spatial, _ = encode_state(game)

    expected_ratio = 2 / 3
    assert spatial[CH_MY_PATTERN, 2, wall_col].item() == pytest.approx(expected_ratio)

    # All other cells in the pattern channel should be 0
    total_nonzero = (spatial[CH_MY_PATTERN] != 0.0).sum().item()
    assert total_nonzero == 1


def test_pattern_line_full_still_shows_ratio_when_wall_empty():
    game = fresh_game()
    current_player = game.state.current_player
    board = game.state.players[current_player]

    # Fill row 1 (capacity 2) with 2 yellow tiles — line is full but wall not yet scored
    yellow = Tile.YELLOW
    board.pattern_lines[1] = [yellow, yellow]
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][1]

    spatial, _ = encode_state(game)
    assert spatial[CH_MY_PATTERN, 1, wall_col].item() == pytest.approx(1.0)


def test_pattern_line_suppressed_when_wall_cell_already_filled():
    game = fresh_game()
    current_player = game.state.current_player
    board = game.state.players[current_player]

    # Put yellow in the pattern line AND the wall cell — wall already scored
    yellow = Tile.YELLOW
    board.pattern_lines[1] = [yellow, yellow]
    wall_col = COLUMN_FOR_TILE_IN_ROW[yellow][1]
    board.wall[1][wall_col] = yellow

    spatial, _ = encode_state(game)
    # Pattern channel should be 0 at that cell — wall already filled
    assert spatial[CH_MY_PATTERN, 1, wall_col].item() == 0.0


# ── Bonus proximity channel ────────────────────────────────────────────────


def test_bonus_proximity_zero_on_empty_wall():
    game = fresh_game()
    spatial, _ = encode_state(game)
    assert spatial[CH_MY_BONUS].sum().item() == 0.0
    assert spatial[CH_OPP_BONUS].sum().item() == 0.0


def test_bonus_proximity_increases_as_wall_fills():
    game = fresh_game()
    current_player = game.state.current_player
    board = game.state.players[current_player]

    spatial_before, _ = encode_state(game)
    proximity_before = spatial_before[CH_MY_BONUS].sum().item()

    # Fill an entire row
    for wall_col in range(BOARD_SIZE):
        color = WALL_PATTERN[0][wall_col]
        board.wall[0][wall_col] = color

    spatial_after, _ = encode_state(game)
    proximity_after = spatial_after[CH_MY_BONUS].sum().item()

    assert proximity_after > proximity_before


def test_bonus_proximity_range():
    game = fresh_game()
    current_player = game.state.current_player
    board = game.state.players[current_player]

    # Fill entire wall
    for row in range(BOARD_SIZE):
        for wall_col in range(BOARD_SIZE):
            board.wall[row][wall_col] = WALL_PATTERN[row][wall_col]

    spatial, _ = encode_state(game)
    values = spatial[CH_MY_BONUS]
    assert values.min().item() >= 0.0
    assert values.max().item() <= 1.0 + 1e-6


# ── Bag count channel ──────────────────────────────────────────────────────


def test_bag_count_broadcast_across_rows():
    game = fresh_game()
    spatial, _ = encode_state(game)

    # Each color's column should have the same value in all rows
    for color_idx in range(BOARD_SIZE):
        col_values = spatial[CH_BAG, :, color_idx]
        assert col_values.min().item() == pytest.approx(col_values.max().item())


def test_bag_count_zero_when_bag_empty():
    game = fresh_game()
    game.state.bag.clear()

    spatial, _ = encode_state(game)
    assert spatial[CH_BAG].sum().item() == 0.0


def test_bag_count_normalized_correctly():
    game = fresh_game()
    game.state.bag.clear()
    # Add 10 blue tiles — blue is COLOR_TILES index 0
    game.state.bag.extend([Tile.BLUE] * 10)

    spatial, _ = encode_state(game)
    blue_col = COLOR_TILES.index(Tile.BLUE)
    expected = 10 / 20
    for row in range(BOARD_SIZE):
        assert spatial[CH_BAG, row, blue_col].item() == pytest.approx(expected)


# ── Source distribution channel ────────────────────────────────────────────


def test_source_distribution_zero_when_no_tiles_available():
    game = fresh_game()
    # Clear all factories and center
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()

    spatial, _ = encode_state(game)
    assert spatial[CH_SOURCE_DIST].sum().item() == 0.0


def test_source_distribution_single_source_with_four_tiles():
    game = fresh_game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()

    # One factory with 4 blue tiles
    blue = Tile.BLUE
    game.state.factories[0].extend([blue] * 4)
    blue_col = COLOR_TILES.index(blue)

    spatial, _ = encode_state(game)

    # Bucket index 3 = sources with 4 tiles; value should be 1/5
    assert spatial[CH_SOURCE_DIST, 3, blue_col].item() == pytest.approx(1 / 5)
    # No other bucket for blue should be nonzero
    for bucket in range(5):
        if bucket != 3:
            assert spatial[CH_SOURCE_DIST, bucket, blue_col].item() == 0.0


def test_source_distribution_bucket_4_catches_five_plus():
    game = fresh_game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()

    # Center with 5 blue tiles — should land in bucket 4 (5+ tiles)
    blue = Tile.BLUE
    game.state.center.extend([blue] * 5)
    blue_col = COLOR_TILES.index(blue)

    spatial, _ = encode_state(game)
    assert spatial[CH_SOURCE_DIST, 4, blue_col].item() == pytest.approx(1 / 5)


def test_source_distribution_two_sources_same_color():
    game = fresh_game()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()

    red = Tile.RED
    game.state.factories[0].extend([red] * 2)
    game.state.factories[1].extend([red] * 2)
    red_col = COLOR_TILES.index(red)

    spatial, _ = encode_state(game)
    # Two sources in bucket 1 (2 tiles each) → value = 2/5
    assert spatial[CH_SOURCE_DIST, 1, red_col].item() == pytest.approx(2 / 5)


# ── Flat vector ────────────────────────────────────────────────────────────


def test_flat_scores_zero_at_game_start():
    game = fresh_game()
    _, flat = encode_state(game)
    # Scores start at 0
    assert flat[OFF_MY_SCORE].item() == 0.0
    assert flat[OFF_OPP_SCORE].item() == 0.0


def test_flat_floor_penalty_negative():
    game = fresh_game()
    current_player = game.state.current_player
    board = game.state.players[current_player]

    # Put 2 tiles on floor — penalty should be -1 + -1 = -2, normalized = -2/14
    board.floor_line.extend([Tile.BLUE, Tile.RED])

    _, flat = encode_state(game)
    assert flat[OFF_MY_FLOOR].item() == pytest.approx(-2 / 14)
    assert flat[OFF_OPP_FLOOR].item() == 0.0


def test_flat_first_player_token_on_my_floor():
    game = fresh_game()
    current_player = game.state.current_player
    board = game.state.players[current_player]
    board.floor_line.append(Tile.FIRST_PLAYER)

    _, flat = encode_state(game)
    assert flat[OFF_MY_FP_TOKEN].item() == 1.0
    assert flat[OFF_OPP_FP_TOKEN].item() == 0.0


def test_flat_first_player_token_on_opponent_floor():
    game = fresh_game()
    opponent = 1 - game.state.current_player
    game.state.players[opponent].floor_line.append(Tile.FIRST_PLAYER)

    _, flat = encode_state(game)
    assert flat[OFF_MY_FP_TOKEN].item() == 0.0
    assert flat[OFF_OPP_FP_TOKEN].item() == 1.0


# ── Move encode / decode roundtrip ────────────────────────────────────────


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
