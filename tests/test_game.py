# tests/test_game.py

"""Tests for core game state dataclasses."""

from engine.game import Tile  # GameState, PlayerBoard,


# ── Tile tests ───────────────────────────────────────────────────────────────


def test_tile_has_six_types():
    assert len(Tile) == 6


def test_tile_has_five_colors():
    color_tiles = [t for t in Tile if t != Tile.FIRST_PLAYER]
    assert len(color_tiles) == 5


def test_tile_first_player_marker_exists():
    assert Tile.FIRST_PLAYER in Tile


# # ── PlayerBoard tests ──────────────────────────────────────────────────────


# def test_player_board_initial_score_is_zero():
#     """A freshly created player board starts with a score of 0."""
#     board = PlayerBoard()
#     assert board.score == 0


# def test_player_board_pattern_lines_has_five_rows():
#     """The pattern lines have 5 rows (row i holds i+1 tiles)."""
#     board = PlayerBoard()
#     assert len(board.pattern_lines) == 5


# def test_player_board_pattern_lines_are_empty():
#     """Each pattern line starts empty (represented as an empty list)."""
#     board = PlayerBoard()
#     for row in board.pattern_lines:
#         assert row == []


# def test_player_board_wall_is_5x5_of_none():
#     """The wall is a 5×5 grid, initially all None (no tiles placed)."""
#     board = PlayerBoard()
#     assert len(board.wall) == 5
#     for row in board.wall:
#         assert len(row) == 5
#         for cell in row:
#             assert cell is None


# def test_player_board_floor_line_is_empty():
#     """The floor line starts empty."""
#     board = PlayerBoard()
#     assert board.floor_line == []


# # ── GameState tests ────────────────────────────────────────────────────────


# def test_game_state_default_has_two_players():
#     """A default GameState is set up for 2 players."""
#     state = GameState()
#     assert len(state.players) == 2


# def test_game_state_current_player_starts_at_zero():
#     """It's player 0's turn at the start of the game."""
#     state = GameState()
#     assert state.current_player == 0


# def test_game_state_two_players_have_five_factories():
#     """A 2-player game uses 5 factory displays."""
#     state = GameState()
#     assert len(state.factories) == 5


# def test_game_state_centre_pool_starts_empty():
#     """The centre pool starts with no tiles (before the bag is drawn)."""
#     state = GameState()
#     assert state.centre == []
