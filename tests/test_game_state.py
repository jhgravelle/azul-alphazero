# tests/test_game.py

"""Tests for core game state dataclasses."""

from engine.game_state import (
    Tile,
    PlayerBoard,
    GameState,
)

from engine.constants import (
    BOARD_SIZE,
    PLAYERS,
    TILES_PER_COLOR,
)

# ── Tile tests ───────────────────────────────────────────────────────────────


def test_tile_has_correct_length():
    assert len(Tile) == BOARD_SIZE + 1  # 5 colors + first-player marker


def test_tile_has_correct_number_of_colors():
    color_tiles = [t for t in Tile if t != Tile.FIRST_PLAYER]
    assert len(color_tiles) == BOARD_SIZE


def test_tile_first_player_marker_exists():
    assert Tile.FIRST_PLAYER in Tile


# # ── PlayerBoard tests ──────────────────────────────────────────────────────


def test_player_board_initial_score_is_zero():
    board = PlayerBoard()
    assert board.score == 0


def test_player_board_pattern_lines_are_empty():
    board = PlayerBoard()
    assert len(board.pattern_lines) == BOARD_SIZE
    for row in board.pattern_lines:
        assert row == []


def test_player_board_wall_is_empty():
    board = PlayerBoard()
    assert len(board.wall) == BOARD_SIZE
    for row in board.wall:
        assert len(row) == BOARD_SIZE
        for cell in row:
            assert cell is None


def test_player_board_floor_line_is_empty():
    """The floor line starts empty."""
    board = PlayerBoard()
    assert board.floor_line == []


# ── GameState tests ────────────────────────────────────────────────────────


def test_game_state_default_players():
    state = GameState()
    assert len(state.players) == PLAYERS


def test_game_state_current_player_starts_at_zero():
    state = GameState()
    assert state.current_player == 0


def test_game_state_factories_are_empty():
    state = GameState()
    assert len(state.factories) == 2 * PLAYERS + 1
    for factory in state.factories:
        assert factory == []


def test_game_state_center_pool_starts_empty():
    state = GameState()
    assert state.center == []


def test_game_state_bag_starts_full():
    state = GameState()
    assert len(state.bag) == BOARD_SIZE * TILES_PER_COLOR


def test_game_state_bag_has_correct_tile_counts():
    state = GameState()
    for tile_type in [t for t in Tile if t != Tile.FIRST_PLAYER]:
        if tile_type == Tile.FIRST_PLAYER:
            assert state.bag.count(tile_type) == 0
        else:
            assert state.bag.count(tile_type) == TILES_PER_COLOR


def test_game_state_bag_is_shuffled():
    state1 = GameState()
    state2 = GameState()
    # It's possible but very unlikely that two shuffles are identical
    assert state1.bag != state2.bag


def test_game_state_discard_is_empty():
    state = GameState()
    assert state.discard == []
