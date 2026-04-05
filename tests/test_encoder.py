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
    STATE_SIZE,
    # Offsets — lets tests pin exact positions without magic numbers
    OFF_MY_WALL,
    OFF_OPP_WALL,
    OFF_MY_PL_FILL,
    OFF_MY_PL_COLOR,
    #    OFF_OPP_PL_FILL,
    OFF_OPP_PL_COLOR,
    OFF_FACTORIES,
    OFF_CENTER,
    OFF_FP_CENTER,
    OFF_FP_MINE,
    OFF_MY_FLOOR,
    OFF_MY_SCORE,
    OFF_OPP_SCORE,
    OFF_BAG,
    OFF_DISCARD,
    OFF_SCORE_DELTA,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    """Return a Game with setup_round() called so there are legal moves."""
    g = Game()
    g.setup_round()
    return g


# ── Shape and type ─────────────────────────────────────────────────────────


def test_state_size_is_157():
    """State vector should be 157 floats after one-hot pattern line colors."""
    assert STATE_SIZE == 157


def test_encode_state_returns_tensor():
    assert isinstance(encode_state(fresh_game()), torch.Tensor)


def test_encode_state_shape():
    assert encode_state(fresh_game()).shape == (STATE_SIZE,)


def test_encode_state_dtype_is_float32():
    assert encode_state(fresh_game()).dtype == torch.float32


def test_encode_state_values_in_range():
    """Most features are in [0, 1]; score delta is in [-1, 1]."""
    t = encode_state(fresh_game())
    assert t.min().item() >= -1.0
    assert t.max().item() <= 1.0


def test_encode_state_non_delta_features_non_negative():
    """All features except the score delta must be >= 0."""
    t = encode_state(fresh_game())
    non_delta = torch.cat([t[:OFF_SCORE_DELTA], t[OFF_SCORE_DELTA + 1 :]])
    assert non_delta.min().item() >= 0.0


# ── Wall ───────────────────────────────────────────────────────────────────


def test_my_wall_empty_at_game_start():
    g = fresh_game()
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_WALL : OFF_MY_WALL + 25].sum().item() == 0.0


def test_my_wall_reflects_current_player_not_player_zero():
    """When current_player=1, my-wall planes must show player 1's wall."""
    g = fresh_game()
    g.state.players[1].wall[0][0] = Tile.BLUE
    g.state.current_player = 1
    t = encode_state(g)
    assert t[OFF_MY_WALL + 0].item() == 1.0


def test_opp_wall_empty_when_only_my_wall_filled():
    g = fresh_game()
    g.state.players[0].wall[2][3] = Tile.RED
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_OPP_WALL : OFF_OPP_WALL + 25].sum().item() == 0.0


def test_my_wall_and_opp_wall_swap_when_current_player_changes():
    g = fresh_game()
    g.state.players[0].wall[0][0] = Tile.BLUE
    g.state.current_player = 0
    t0 = encode_state(g)
    g.state.current_player = 1
    t1 = encode_state(g)
    assert t0[OFF_MY_WALL + 0].item() == 1.0
    assert t0[OFF_OPP_WALL + 0].item() == 0.0
    assert t1[OFF_MY_WALL + 0].item() == 0.0
    assert t1[OFF_OPP_WALL + 0].item() == 1.0


# ── Pattern lines ──────────────────────────────────────────────────────────


def test_my_pattern_line_fill_ratio_empty():
    g = fresh_game()
    t = encode_state(g)
    assert t[OFF_MY_PL_FILL : OFF_MY_PL_FILL + BOARD_SIZE].sum().item() == 0.0


def test_my_pattern_line_fill_ratio_partial():
    """Row 2 (capacity 3) with 2 tiles should encode as 2/3."""
    g = fresh_game()
    g.state.players[0].pattern_lines[2] = [Tile.RED, Tile.RED]
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_PL_FILL + 2].item() == pytest.approx(2 / 3)


def test_my_pattern_line_fill_ratio_full():
    """Row 4 (capacity 5) fully filled should encode as 1.0."""
    g = fresh_game()
    g.state.players[0].pattern_lines[4] = [Tile.WHITE] * 5
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_PL_FILL + 4].item() == pytest.approx(1.0)


def test_my_pattern_line_color_empty_is_all_zeros():
    """An empty pattern line should have all zeros for its color one-hot."""
    g = fresh_game()
    g.state.current_player = 0
    t = encode_state(g)
    # All 25 color floats (5 rows × 5 colors) should be zero
    assert t[OFF_MY_PL_COLOR : OFF_MY_PL_COLOR + 25].sum().item() == 0.0


def test_my_pattern_line_color_is_one_hot():
    """A pattern line with Blue tiles should have a one-hot at the Blue index."""
    g = fresh_game()
    g.state.players[0].pattern_lines[0] = [Tile.BLUE]
    g.state.current_player = 0
    t = encode_state(g)

    blue_idx = COLOR_TILES.index(Tile.BLUE)
    for i in range(BOARD_SIZE):
        if i == blue_idx:
            assert t[OFF_MY_PL_COLOR + 0 * BOARD_SIZE + i] == 1.0
        else:
            assert t[OFF_MY_PL_COLOR + 0 * BOARD_SIZE + i] == 0.0


def test_my_pattern_line_color_each_color_distinct():
    """Each color should activate a different position in the one-hot vector."""
    g = fresh_game()
    g.state.current_player = 0
    hot_positions = set()
    for i, color in enumerate(COLOR_TILES):
        g.state.players[0].pattern_lines[i] = [color]
        t = encode_state(g)
        # Find which of the 5 floats is 1.0 for this row
        row_start = OFF_MY_PL_COLOR + i * BOARD_SIZE
        for j in range(BOARD_SIZE):
            if t[row_start + j].item() == 1.0:
                hot_positions.add(j)
        g.state.players[0].pattern_lines[i] = []
    assert len(hot_positions) == BOARD_SIZE


def test_opp_pattern_line_color_is_one_hot():
    """Opponent pattern line colors should also be one-hot encoded."""
    g = fresh_game()
    g.state.players[1].pattern_lines[2] = [Tile.RED, Tile.RED]
    g.state.current_player = 0
    t = encode_state(g)

    red_idx = COLOR_TILES.index(Tile.RED)
    for i in range(BOARD_SIZE):
        if i == red_idx:
            assert t[OFF_OPP_PL_COLOR + 2 * BOARD_SIZE + i] == 1.0
        else:
            assert t[OFF_OPP_PL_COLOR + 2 * BOARD_SIZE + i] == 0.0


# ── Factories ──────────────────────────────────────────────────────────────


def test_factories_all_zero_before_setup():
    g = Game()
    t = encode_state(g)
    assert t[OFF_FACTORIES : OFF_FACTORIES + 25].sum().item() == 0.0


def test_factories_nonzero_after_setup():
    t = encode_state(fresh_game())
    assert t[OFF_FACTORIES : OFF_FACTORIES + 25].sum().item() > 0.0


def test_factory_color_count_normalized():
    """A factory with 4 tiles of one color should encode that color as 1.0."""
    g = Game()
    g.state.factories[0] = [Tile.BLUE] * 4
    t = encode_state(g)
    blue_idx = COLOR_TILES.index(Tile.BLUE)
    assert t[OFF_FACTORIES + 0 * BOARD_SIZE + blue_idx].item() == pytest.approx(1.0)


# ── Center ─────────────────────────────────────────────────────────────────


def test_center_all_zero_before_setup():
    g = Game()
    t = encode_state(g)
    assert t[OFF_CENTER : OFF_CENTER + BOARD_SIZE].sum().item() == 0.0


def test_center_color_count_correct():
    g = Game()
    g.state.center = [Tile.RED, Tile.RED, Tile.RED]
    t = encode_state(g)
    red_idx = COLOR_TILES.index(Tile.RED)
    assert t[OFF_CENTER + red_idx].item() == pytest.approx(3 / TILES_PER_COLOR)


# ── First player token ─────────────────────────────────────────────────────


def test_first_player_token_in_center_when_present():
    g = fresh_game()
    g.state.center.append(Tile.FIRST_PLAYER)
    t = encode_state(g)
    assert t[OFF_FP_CENTER].item() == 1.0


def test_first_player_token_not_in_center_when_absent():
    g = fresh_game()
    g.state.center = [tile for tile in g.state.center if tile != Tile.FIRST_PLAYER]
    t = encode_state(g)
    assert t[OFF_FP_CENTER].item() == 0.0


def test_first_player_token_mine_when_on_my_floor():
    g = fresh_game()
    g.state.players[0].floor_line = [Tile.FIRST_PLAYER]
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_FP_MINE].item() == 1.0


def test_first_player_token_not_mine_when_on_opponent_floor():
    g = fresh_game()
    g.state.players[1].floor_line = [Tile.FIRST_PLAYER]
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_FP_MINE].item() == 0.0


# ── Floor ──────────────────────────────────────────────────────────────────


def test_my_floor_zero_when_empty():
    g = fresh_game()
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_FLOOR].item() == 0.0


def test_my_floor_normalized():
    g = fresh_game()
    g.state.players[0].floor_line = [Tile.BLUE, Tile.RED, Tile.BLACK]
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_FLOOR].item() == pytest.approx(3 / 7)


# ── Score ──────────────────────────────────────────────────────────────────


def test_my_score_zero_at_start():
    g = fresh_game()
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_SCORE].item() == 0.0


def test_my_score_normalized():
    g = fresh_game()
    g.state.players[0].score = 50
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_SCORE].item() == pytest.approx(50 / 100)


def test_scores_swap_with_current_player():
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 60
    g.state.current_player = 0
    t0 = encode_state(g)
    g.state.current_player = 1
    t1 = encode_state(g)
    assert t0[OFF_MY_SCORE].item() == pytest.approx(30 / 100)
    assert t0[OFF_OPP_SCORE].item() == pytest.approx(60 / 100)
    assert t1[OFF_MY_SCORE].item() == pytest.approx(60 / 100)
    assert t1[OFF_OPP_SCORE].item() == pytest.approx(30 / 100)


# ── Score delta ────────────────────────────────────────────────────────────


def test_score_delta_zero_when_scores_equal():
    g = fresh_game()
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_SCORE_DELTA].item() == pytest.approx(0.0)


def test_score_delta_positive_when_ahead():
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 10
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_SCORE_DELTA].item() == pytest.approx((30 - 10) / 20)


def test_score_delta_negative_when_behind():
    g = fresh_game()
    g.state.players[0].score = 10
    g.state.players[1].score = 30
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_SCORE_DELTA].item() == pytest.approx((10 - 30) / 20)


def test_score_delta_clamped_at_positive_one():
    g = fresh_game()
    g.state.players[0].score = 100
    g.state.players[1].score = 0
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_SCORE_DELTA].item() == pytest.approx(1.0)


def test_score_delta_clamped_at_negative_one():
    g = fresh_game()
    g.state.players[0].score = 0
    g.state.players[1].score = 100
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_SCORE_DELTA].item() == pytest.approx(-1.0)


def test_score_delta_flips_with_current_player():
    """Delta from player 1's perspective is the negative of player 0's."""
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 10
    g.state.current_player = 0
    t0 = encode_state(g)
    g.state.current_player = 1
    t1 = encode_state(g)
    assert t0[OFF_SCORE_DELTA].item() == pytest.approx(-t1[OFF_SCORE_DELTA].item())


def test_score_delta_is_at_last_offset():
    """Score delta must be the last feature in the state vector."""
    assert OFF_SCORE_DELTA == STATE_SIZE - 1


# ── Bag and discard ────────────────────────────────────────────────────────


def test_bag_full_at_start():
    g = Game()
    t = encode_state(g)
    assert t[OFF_BAG : OFF_BAG + BOARD_SIZE].sum().item() == pytest.approx(
        BOARD_SIZE * 1.0
    )


def test_bag_decreases_after_setup():
    t = encode_state(fresh_game())
    assert t[OFF_BAG : OFF_BAG + BOARD_SIZE].sum().item() < BOARD_SIZE * 1.0


def test_discard_zero_at_start():
    g = Game()
    t = encode_state(g)
    assert t[OFF_DISCARD : OFF_DISCARD + BOARD_SIZE].sum().item() == 0.0


def test_bag_and_discard_correct_counts():
    g = Game()
    g.state.bag = [Tile.BLUE] * 10 + [Tile.RED] * 5
    g.state.discard = [Tile.YELLOW] * 8
    t = encode_state(g)
    blue_idx = COLOR_TILES.index(Tile.BLUE)
    red_idx = COLOR_TILES.index(Tile.RED)
    yellow_idx = COLOR_TILES.index(Tile.YELLOW)
    assert t[OFF_BAG + blue_idx].item() == pytest.approx(10 / TILES_PER_COLOR)
    assert t[OFF_BAG + red_idx].item() == pytest.approx(5 / TILES_PER_COLOR)
    assert t[OFF_DISCARD + yellow_idx].item() == pytest.approx(8 / TILES_PER_COLOR)


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
