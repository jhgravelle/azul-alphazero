# tests/test_game.py

"""Tests for core game methods."""

from engine.game import CENTER, Game
from engine.constants import TILES_PER_FACTORY
from engine.game_state import Tile


def test_setup_round_fills_factories():
    game = Game()
    game.setup_round()
    for factory in game.state.factories:
        assert len(factory) == TILES_PER_FACTORY


def test_setup_round_draws_from_bag():
    game = Game()
    initial_bag_size = len(game.state.bag)
    game.setup_round()
    num_factories = len(game.state.factories)
    expected_tiles_drawn = num_factories * TILES_PER_FACTORY
    assert len(game.state.bag) == initial_bag_size - expected_tiles_drawn


def test_setup_round_uses_discard_when_bag_empty():
    game = Game()
    game.state.discard = game.state.bag.copy()
    game.state.bag.clear()
    game.setup_round()
    for factory in game.state.factories:
        assert len(factory) == TILES_PER_FACTORY


def test_setup_round_partial_fill_when_no_tiles():
    game = Game()
    game.state.bag.clear()
    game.state.discard.clear()
    game.setup_round()
    total_tiles = sum(len(f) for f in game.state.factories)
    assert total_tiles == 0


def test_setup_round_fills_factories_in_order():
    game = Game()
    game.state.bag = [Tile.BLUE] * (TILES_PER_FACTORY + 1)
    game.state.discard.clear()
    game.setup_round()
    assert len(game.state.factories[0]) == TILES_PER_FACTORY
    assert len(game.state.factories[1]) == 1
    for factory in game.state.factories[2:]:
        assert len(factory) == 0


def test_legal_moves_returns_moves_after_setup():
    game = Game()
    game.setup_round()
    moves = game.legal_moves()
    assert len(moves) > 0


def test_legal_moves_only_contain_colors_present_in_source():
    game = Game()
    game.setup_round()
    for move in game.legal_moves():
        if move.source == CENTER:
            source_tiles = game.state.center
        else:
            source_tiles = game.state.factories[move.source]
        assert move.color in source_tiles


def test_legal_moves_empty_center_generates_no_center_moves():
    game = Game()
    game.setup_round()
    game.state.center.clear()  # center starts empty after setup_round
    center_moves = [m for m in game.legal_moves() if m.source == CENTER]
    assert len(center_moves) == 0


def test_legal_moves_exclude_pattern_line_with_different_color():
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    player.pattern_lines[0].append(Tile.BLUE)
    moves = game.legal_moves()
    invalid_moves = [m for m in moves if m.destination == 0 and m.color != Tile.BLUE]
    assert len(invalid_moves) == 0


def test_legal_moves_exclude_full_pattern_line():
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    player.pattern_lines[0] = [Tile.BLUE]  # row 0 holds max 1 tile, so it's full
    moves = game.legal_moves()
    invalid_moves = [m for m in moves if m.destination == 0]
    assert len(invalid_moves) == 0


def test_legal_moves_exclude_pattern_line_where_wall_row_has_color():
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    player.wall[0][0] = Tile.BLUE  # place blue on row 0 of the wall
    moves = game.legal_moves()
    invalid_moves = [m for m in moves if m.destination == 0 and m.color == Tile.BLUE]
    assert len(invalid_moves) == 0
