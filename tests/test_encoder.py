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
    OFF_FACTORIES,
    OFF_CENTER,
    OFF_FP_CENTER,
    OFF_FP_MINE,
    OFF_MY_FLOOR,
    OFF_MY_SCORE,
    OFF_OPP_SCORE,
    OFF_BAG,
    OFF_DISCARD,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def fresh_game() -> Game:
    """Return a Game with setup_round() called so there are legal moves."""
    g = Game()
    g.setup_round()
    return g


# ── Shape and type ─────────────────────────────────────────────────────────


def test_encode_state_returns_tensor():
    assert isinstance(encode_state(fresh_game()), torch.Tensor)


def test_encode_state_shape():
    assert encode_state(fresh_game()).shape == (STATE_SIZE,)


def test_encode_state_dtype_is_float32():
    assert encode_state(fresh_game()).dtype == torch.float32


def test_encode_state_values_in_range():
    t = encode_state(fresh_game())
    assert t.min().item() >= 0.0
    assert t.max().item() <= 1.0


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
    # Cell (row=0, col=0) of my wall — flat index = OFF_MY_WALL + 0*5 + 0
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
    # From player 0's perspective it's my wall; from player 1's it's opp wall
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


def test_my_pattern_line_color_empty_is_zero():
    g = fresh_game()
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_PL_COLOR : OFF_MY_PL_COLOR + BOARD_SIZE].sum().item() == 0.0


def test_my_pattern_line_color_nonzero_when_filled():
    """Any non-empty pattern line must produce a color value > 0."""
    g = fresh_game()
    g.state.players[0].pattern_lines[1] = [Tile.YELLOW]
    g.state.current_player = 0
    t = encode_state(g)
    assert t[OFF_MY_PL_COLOR + 1].item() > 0.0


def test_my_pattern_line_color_unique_per_color():
    """Each color must produce a distinct encoded value."""
    g = fresh_game()
    g.state.current_player = 0
    values = set()
    for i, color in enumerate(COLOR_TILES):
        g.state.players[0].pattern_lines[i] = [color]
        t = encode_state(g)
        val = t[OFF_MY_PL_COLOR + i].item()
        assert val != 0.0
        values.add(round(val, 6))
        g.state.players[0].pattern_lines[i] = []  # reset
    assert len(values) == BOARD_SIZE


# ── Factories ──────────────────────────────────────────────────────────────


def test_factories_all_zero_before_setup():
    g = Game()  # no setup_round
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


# ── Bag and discard ────────────────────────────────────────────────────────


def test_bag_full_at_start():
    g = Game()
    t = encode_state(g)
    # Each color starts with TILES_PER_COLOR tiles → normalized = 1.0
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
        assert recovered.color == move.color
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
