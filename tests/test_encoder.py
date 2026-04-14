# tests/test_encoder.py
"""Tests for the neural network state and move encoder."""

import torch
import pytest
from engine.game import Game, CENTER, FLOOR
from engine.constants import Tile, BOARD_SIZE, COLOR_TILES, TILES_PER_COLOR
from neural.encoder import (
    encode_state,
    encode_move,
    decode_move,
    MOVE_SPACE_SIZE,
    SPATIAL_SHAPE,
    FLAT_SIZE,
    NUM_COLORS,
    PLANES_PER_PLAYER,
    PATTERN_COL,
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
)

# ── Helpers ────────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    g = Game()
    g.setup_round()
    return g


def encode(g: Game):
    """Return (spatial, flat) for convenience."""
    return encode_state(g)


# ── Return types and shapes ────────────────────────────────────────────────


def test_encode_state_returns_tuple():
    spatial, flat = encode(fresh_game())
    assert isinstance(spatial, torch.Tensor)
    assert isinstance(flat, torch.Tensor)


def test_spatial_shape():
    spatial, _ = encode(fresh_game())
    assert spatial.shape == SPATIAL_SHAPE  # (12, 5, 6)


def test_flat_shape():
    _, flat = encode(fresh_game())
    assert flat.shape == (FLAT_SIZE,)  # (47,)


def test_spatial_dtype():
    spatial, _ = encode(fresh_game())
    assert spatial.dtype == torch.float32


def test_flat_dtype():
    _, flat = encode(fresh_game())
    assert flat.dtype == torch.float32


def test_spatial_values_in_range():
    spatial, _ = encode(fresh_game())
    assert spatial.min().item() >= 0.0
    assert spatial.max().item() <= 1.0


def test_flat_values_in_range():
    """Most flat features are in [0,1]; score delta is in [-1, 1]."""
    _, flat = encode(fresh_game())
    assert flat.min().item() >= -1.0
    assert flat.max().item() <= 1.0


def test_flat_non_delta_features_non_negative():
    _, flat = encode(fresh_game())
    non_delta = torch.cat([flat[:OFF_SCORE_DELTA], flat[OFF_SCORE_DELTA + 1 :]])
    assert non_delta.min().item() >= 0.0


# ── Spatial: wall cells ────────────────────────────────────────────────────


def test_my_wall_channels_empty_at_game_start():
    """All 6 current-player channels should be zero before any tiles placed."""
    g = fresh_game()
    g.state.current_player = 0
    spatial, _ = encode(g)
    assert spatial[:PLANES_PER_PLAYER, :, :BOARD_SIZE].sum().item() == 0.0


def test_my_wall_reflects_current_player_not_player_zero():
    """When current_player=1, my-channels must show player 1's wall."""
    g = fresh_game()
    g.state.players[1].wall[0][0] = Tile.BLUE
    g.state.current_player = 1
    spatial, _ = encode(g)
    blue_idx = COLOR_TILES.index(Tile.BLUE)
    assert spatial[blue_idx, 0, 0].item() == 1.0


def test_my_wall_color_plane_set_correctly():
    g = fresh_game()
    g.state.players[0].wall[2][3] = Tile.RED
    g.state.current_player = 0
    spatial, _ = encode(g)
    red_idx = COLOR_TILES.index(Tile.RED)
    assert spatial[red_idx, 2, 3].item() == 1.0


def test_my_wall_any_plane_set_when_tile_present():
    g = fresh_game()
    g.state.players[0].wall[1][4] = Tile.BLACK
    g.state.current_player = 0
    spatial, _ = encode(g)
    any_channel = NUM_COLORS  # channel 5
    assert spatial[any_channel, 1, 4].item() == 1.0


def test_opp_wall_channels_zero_when_only_my_wall_filled():
    g = fresh_game()
    g.state.players[0].wall[2][3] = Tile.RED
    g.state.current_player = 0
    spatial, _ = encode(g)
    opp_channels = spatial[PLANES_PER_PLAYER:, :, :BOARD_SIZE]
    assert opp_channels.sum().item() == 0.0


def test_my_and_opp_wall_swap_when_current_player_changes():
    g = fresh_game()
    g.state.players[0].wall[0][0] = Tile.BLUE
    blue_idx = COLOR_TILES.index(Tile.BLUE)

    g.state.current_player = 0
    s0, _ = encode(g)
    g.state.current_player = 1
    s1, _ = encode(g)

    # As player 0: my channel has the tile
    assert s0[blue_idx, 0, 0].item() == 1.0
    assert s0[PLANES_PER_PLAYER + blue_idx, 0, 0].item() == 0.0

    # As player 1: opp channel has the tile
    assert s1[blue_idx, 0, 0].item() == 0.0
    assert s1[PLANES_PER_PLAYER + blue_idx, 0, 0].item() == 1.0


def test_only_correct_color_channel_set_for_wall_tile():
    """When a Red tile is placed, only the Red channel should be nonzero at that "
    "cell."""
    g = fresh_game()
    g.state.players[0].wall[3][2] = Tile.RED
    g.state.current_player = 0
    spatial, _ = encode(g)
    red_idx = COLOR_TILES.index(Tile.RED)
    for c in range(NUM_COLORS):
        val = spatial[c, 3, 2].item()
        if c == red_idx:
            assert val == 1.0
        else:
            assert val == 0.0


# ── Spatial: pattern line column ───────────────────────────────────────────


def test_pattern_line_col_zero_when_empty():
    g = fresh_game()
    g.state.current_player = 0
    spatial, _ = encode(g)
    assert spatial[:PLANES_PER_PLAYER, :, PATTERN_COL].sum().item() == 0.0


def test_pattern_line_fill_ratio_partial():
    """Row 2 (capacity 3) with 2 Red tiles → Red channel col 5 = 2/3."""
    g = fresh_game()
    g.state.players[0].pattern_lines[2] = [Tile.RED, Tile.RED]
    g.state.current_player = 0
    spatial, _ = encode(g)
    red_idx = COLOR_TILES.index(Tile.RED)
    assert spatial[red_idx, 2, PATTERN_COL].item() == pytest.approx(2 / 3)


def test_pattern_line_fill_ratio_full():
    """Row 4 (capacity 5) fully filled → 1.0."""
    g = fresh_game()
    g.state.players[0].pattern_lines[4] = [Tile.WHITE] * 5
    g.state.current_player = 0
    spatial, _ = encode(g)
    white_idx = COLOR_TILES.index(Tile.WHITE)
    assert spatial[white_idx, 4, PATTERN_COL].item() == pytest.approx(1.0)


def test_pattern_line_only_correct_color_channel_nonzero():
    """A Blue pattern line should set only the Blue channel in col 5."""
    g = fresh_game()
    g.state.players[0].pattern_lines[1] = [Tile.BLUE, Tile.BLUE]
    g.state.current_player = 0
    spatial, _ = encode(g)
    blue_idx = COLOR_TILES.index(Tile.BLUE)
    for c in range(NUM_COLORS):
        val = spatial[c, 1, PATTERN_COL].item()
        if c == blue_idx:
            assert val == pytest.approx(1.0)
        else:
            assert val == 0.0


def test_pattern_line_any_channel_matches_fill_ratio():
    """The any-tile channel in col 5 should equal the fill ratio."""
    g = fresh_game()
    g.state.players[0].pattern_lines[3] = [Tile.YELLOW] * 2
    g.state.current_player = 0
    spatial, _ = encode(g)
    any_channel = NUM_COLORS
    assert spatial[any_channel, 3, PATTERN_COL].item() == pytest.approx(2 / 4)


def test_opp_pattern_line_encoded_in_opp_channels():
    """Opponent's pattern line should appear in channels 6–11, not 0–5."""
    g = fresh_game()
    g.state.players[1].pattern_lines[0] = [Tile.RED]
    g.state.current_player = 0
    spatial, _ = encode(g)
    red_idx = COLOR_TILES.index(Tile.RED)
    # My channels should be zero
    assert spatial[red_idx, 0, PATTERN_COL].item() == 0.0
    # Opp channels should be set
    assert spatial[PLANES_PER_PLAYER + red_idx, 0, PATTERN_COL].item() == pytest.approx(
        1.0
    )


def test_pattern_line_different_color_from_wall_independent():
    """Pattern line and wall cell on the same row can be different colors."""
    g = fresh_game()
    g.state.players[0].wall[0][1] = Tile.RED
    g.state.players[0].pattern_lines[0] = [Tile.BLUE]
    g.state.current_player = 0
    spatial, _ = encode(g)
    red_idx = COLOR_TILES.index(Tile.RED)
    blue_idx = COLOR_TILES.index(Tile.BLUE)
    assert spatial[red_idx, 0, 1].item() == 1.0  # wall cell
    assert spatial[blue_idx, 0, PATTERN_COL].item() == pytest.approx(
        1.0
    )  # pattern line


# ── Flat: factories ────────────────────────────────────────────────────────


def test_factories_all_zero_before_setup():
    g = Game()
    _, flat = encode(g)
    assert flat[OFF_FACTORIES : OFF_FACTORIES + 25].sum().item() == 0.0


def test_factories_nonzero_after_setup():
    _, flat = encode(fresh_game())
    assert flat[OFF_FACTORIES : OFF_FACTORIES + 25].sum().item() > 0.0


def test_factory_color_count_normalized():
    """A factory with 4 Blue tiles should encode that color as 1.0."""
    g = Game()
    g.state.factories[0] = [Tile.BLUE] * 4
    _, flat = encode(g)
    blue_idx = COLOR_TILES.index(Tile.BLUE)
    assert flat[OFF_FACTORIES + 0 * NUM_COLORS + blue_idx].item() == pytest.approx(1.0)


# ── Flat: center ───────────────────────────────────────────────────────────


def test_center_all_zero_before_setup():
    g = Game()
    _, flat = encode(g)
    assert flat[OFF_CENTER : OFF_CENTER + BOARD_SIZE].sum().item() == 0.0


def test_center_color_count_correct():
    g = Game()
    g.state.center = [Tile.RED, Tile.RED, Tile.RED]
    _, flat = encode(g)
    red_idx = COLOR_TILES.index(Tile.RED)
    assert flat[OFF_CENTER + red_idx].item() == pytest.approx(3 / TILES_PER_COLOR)


# ── Flat: first player token ───────────────────────────────────────────────


def test_first_player_token_in_center_when_present():
    g = fresh_game()
    g.state.center.append(Tile.FIRST_PLAYER)
    _, flat = encode(g)
    assert flat[OFF_FP_CENTER].item() == 1.0


def test_first_player_token_not_in_center_when_absent():
    g = fresh_game()
    g.state.center = [t for t in g.state.center if t != Tile.FIRST_PLAYER]
    _, flat = encode(g)
    assert flat[OFF_FP_CENTER].item() == 0.0


def test_first_player_token_mine_when_on_my_floor():
    g = fresh_game()
    g.state.players[0].floor_line = [Tile.FIRST_PLAYER]
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_FP_MINE].item() == 1.0


def test_first_player_token_not_mine_when_on_opponent_floor():
    g = fresh_game()
    g.state.players[1].floor_line = [Tile.FIRST_PLAYER]
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_FP_MINE].item() == 0.0


# ── Flat: floor ────────────────────────────────────────────────────────────


def test_my_floor_zero_when_empty():
    g = fresh_game()
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_MY_FLOOR].item() == 0.0


def test_my_floor_normalized():
    g = fresh_game()
    g.state.players[0].floor_line = [Tile.BLUE, Tile.RED, Tile.BLACK]
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_MY_FLOOR].item() == pytest.approx(3 / 7)


def test_opp_floor_reflects_opponent():
    g = fresh_game()
    g.state.players[1].floor_line = [Tile.YELLOW] * 4
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_OPP_FLOOR].item() == pytest.approx(4 / 7)


def test_floor_swaps_with_current_player():
    g = fresh_game()
    g.state.players[0].floor_line = [Tile.BLUE]
    g.state.players[1].floor_line = [Tile.RED, Tile.RED]
    g.state.current_player = 0
    _, f0 = encode(g)
    g.state.current_player = 1
    _, f1 = encode(g)
    assert f0[OFF_MY_FLOOR].item() == pytest.approx(1 / 7)
    assert f0[OFF_OPP_FLOOR].item() == pytest.approx(2 / 7)
    assert f1[OFF_MY_FLOOR].item() == pytest.approx(2 / 7)
    assert f1[OFF_OPP_FLOOR].item() == pytest.approx(1 / 7)


# ── Flat: scores ───────────────────────────────────────────────────────────


def test_my_score_zero_at_start():
    g = fresh_game()
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_MY_SCORE].item() == 0.0


def test_scores_swap_with_current_player():
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 60
    g.state.current_player = 0
    _, f0 = encode(g)
    g.state.current_player = 1
    _, f1 = encode(g)
    assert f0[OFF_MY_SCORE].item() == pytest.approx(30 / 100)
    assert f0[OFF_OPP_SCORE].item() == pytest.approx(60 / 100)
    assert f1[OFF_MY_SCORE].item() == pytest.approx(60 / 100)
    assert f1[OFF_OPP_SCORE].item() == pytest.approx(30 / 100)


# ── Flat: score delta ──────────────────────────────────────────────────────


def test_score_delta_zero_when_equal():
    g = fresh_game()
    _, flat = encode(g)
    assert flat[OFF_SCORE_DELTA].item() == pytest.approx(0.0)


def test_score_delta_positive_when_ahead():
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 10
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_SCORE_DELTA].item() == pytest.approx((30 - 10) / 20)


def test_score_delta_clamped_at_positive_one():
    g = fresh_game()
    g.state.players[0].score = 100
    g.state.players[1].score = 0
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_SCORE_DELTA].item() == pytest.approx(1.0)


def test_score_delta_clamped_at_negative_one():
    g = fresh_game()
    g.state.players[0].score = 0
    g.state.players[1].score = 100
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_SCORE_DELTA].item() == pytest.approx(-1.0)


def test_score_delta_flips_with_current_player():
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 10
    g.state.current_player = 0
    _, f0 = encode(g)
    g.state.current_player = 1
    _, f1 = encode(g)
    assert f0[OFF_SCORE_DELTA].item() == pytest.approx(-f1[OFF_SCORE_DELTA].item())


# ── Flat: bag and discard ──────────────────────────────────────────────────


def test_bag_full_at_start():
    g = Game()
    _, flat = encode(g)
    assert flat[OFF_BAG : OFF_BAG + NUM_COLORS].sum().item() == pytest.approx(
        NUM_COLORS * 1.0
    )


def test_bag_decreases_after_setup():
    _, flat = encode(fresh_game())
    assert flat[OFF_BAG : OFF_BAG + NUM_COLORS].sum().item() < NUM_COLORS * 1.0


def test_discard_zero_at_start():
    g = Game()
    _, flat = encode(g)
    assert flat[OFF_DISCARD : OFF_DISCARD + NUM_COLORS].sum().item() == 0.0


def test_bag_and_discard_correct_counts():
    g = Game()
    g.state.bag = [Tile.BLUE] * 10 + [Tile.RED] * 5
    g.state.discard = [Tile.YELLOW] * 8
    _, flat = encode(g)
    blue_idx = COLOR_TILES.index(Tile.BLUE)
    red_idx = COLOR_TILES.index(Tile.RED)
    yellow_idx = COLOR_TILES.index(Tile.YELLOW)
    assert flat[OFF_BAG + blue_idx].item() == pytest.approx(10 / TILES_PER_COLOR)
    assert flat[OFF_BAG + red_idx].item() == pytest.approx(5 / TILES_PER_COLOR)
    assert flat[OFF_DISCARD + yellow_idx].item() == pytest.approx(8 / TILES_PER_COLOR)


# ── Move encoding ──────────────────────────────────────────────────────────


def test_move_space_size_is_positive():
    assert MOVE_SPACE_SIZE > 0


def test_encode_move_returns_int():
    g = fresh_game()
    assert isinstance(encode_move(g.legal_moves()[0], g), int)


def test_encode_move_index_in_range():
    g = fresh_game()
    for move in g.legal_moves():
        idx = encode_move(move, g)
        assert 0 <= idx < MOVE_SPACE_SIZE, f"Index {idx} out of range for {move}"


def test_encode_decode_move_roundtrip():
    g = fresh_game()
    for move in g.legal_moves():
        recovered = decode_move(encode_move(move, g), g)
        assert recovered.source == move.source
        assert recovered.tile == move.tile
        assert recovered.destination == move.destination


def test_encode_move_is_deterministic():
    g = fresh_game()
    move = g.legal_moves()[0]
    assert encode_move(move, g) == encode_move(move, g)


def test_encode_move_different_moves_have_different_indices():
    g = fresh_game()
    moves = g.legal_moves()
    if len(moves) < 2:
        pytest.skip("need at least 2 legal moves")
    indices = [encode_move(m, g) for m in moves]
    assert len(indices) == len(set(indices))


def test_decode_move_floor_destination():
    g = fresh_game()
    floor_moves = [m for m in g.legal_moves() if m.destination == FLOOR]
    assert floor_moves
    for move in floor_moves:
        assert decode_move(encode_move(move, g), g).destination == FLOOR


def test_decode_move_center_source():
    g = Game()
    g.setup_round()
    g.state.center.append(Tile.BLUE)
    center_moves = [m for m in g.legal_moves() if m.source == CENTER]
    if not center_moves:
        pytest.skip("no center moves available")
    for move in center_moves:
        assert decode_move(encode_move(move, g), g).source == CENTER
