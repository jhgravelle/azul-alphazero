# tests/test_game.py

"""Tests for core game methods."""

from engine.game import CENTER, FLOOR, Game, Move
from engine.constants import (
    Tile,
    BOARD_SIZE,
    COLOR_TILES,
    TILES_PER_FACTORY,
    WALL_PATTERN,
)


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


def test_wall_pattern_is_correct_size():
    assert len(WALL_PATTERN) == BOARD_SIZE
    for row in WALL_PATTERN:
        assert len(row) == BOARD_SIZE


def test_wall_pattern_each_row_and_column_has_all_colors():
    colors = set([t for t in Tile if t != Tile.FIRST_PLAYER])
    for i in range(BOARD_SIZE):
        assert set(WALL_PATTERN[i]) == colors, f"Row {i} missing colors"
        col = {WALL_PATTERN[r][i] for r in range(BOARD_SIZE)}
        assert col == colors, f"Column {i} missing colors"


def test_wall_pattern_known_positions():
    # Row 0: Blue Yellow Red Black White
    assert WALL_PATTERN[0][0] == Tile.BLUE
    assert WALL_PATTERN[0][4] == Tile.WHITE
    # Row 1 shifts left by one: White Blue Yellow Red Black
    assert WALL_PATTERN[1][0] == Tile.WHITE
    assert WALL_PATTERN[1][1] == Tile.BLUE
    # Row 4: Yellow Red Black White Blue
    assert WALL_PATTERN[4][0] == Tile.YELLOW
    assert WALL_PATTERN[4][4] == Tile.BLUE


def test_wall_column_for_known_positions():
    game = Game()
    assert game.wall_column_for(row=0, color=Tile.BLUE) == 0
    assert game.wall_column_for(row=0, color=Tile.WHITE) == 4
    assert game.wall_column_for(row=1, color=Tile.BLUE) == 1
    assert game.wall_column_for(row=4, color=Tile.BLUE) == 4


def test_wall_column_for_every_row_returns_unique_columns():
    game = Game()
    for row in range(BOARD_SIZE):
        cols = [game.wall_column_for(row=row, color=c) for c in COLOR_TILES]
        assert sorted(cols) == list(
            range(BOARD_SIZE)
        ), f"Row {row} columns not unique: {cols}"


# --- make_move tests ---


def test_make_move_removes_color_from_factory():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.make_move(Move(source=0, color=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.state.factories[0]


def test_make_move_leftover_factory_tiles_go_to_center():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.make_move(Move(source=0, color=Tile.BLUE, destination=0))
    assert Tile.RED in game.state.center
    assert Tile.YELLOW in game.state.center


def test_make_move_taking_from_center_leaves_no_leftover():
    game = Game()
    game.state.center = [Tile.BLUE, Tile.BLUE, Tile.RED]
    game.make_move(Move(source=CENTER, color=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.state.center
    assert Tile.RED in game.state.center


def test_make_move_taking_from_center_moves_first_player_marker_to_floor():
    game = Game()
    game.state.center = [
        Tile.FIRST_PLAYER,
        Tile.BLUE,
        Tile.BLUE,
        Tile.RED,
    ]  # making sure the round doesn't end
    game.make_move(Move(source=CENTER, color=Tile.BLUE, destination=0))
    player = game.state.players[game.state.current_player - 1]
    assert Tile.FIRST_PLAYER in player.floor_line
    assert Tile.FIRST_PLAYER not in game.state.center


def test_make_move_places_tiles_on_pattern_line():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.make_move(Move(source=0, color=Tile.BLUE, destination=1))
    player = game.state.players[0]
    assert player.pattern_lines[1].count(Tile.BLUE) == 2


def test_make_move_overflow_tiles_go_to_floor():
    game = Game()
    # Row 0 holds max 1 tile — sending 2 blues there should overflow 1
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.make_move(Move(source=0, color=Tile.BLUE, destination=0))
    player = game.state.players[0]
    assert player.pattern_lines[0] == [Tile.BLUE]
    assert player.floor_line.count(Tile.BLUE) == 1


def test_make_move_to_floor_puts_all_tiles_on_floor():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.make_move(Move(source=0, color=Tile.BLUE, destination=FLOOR))
    player = game.state.players[0]
    assert player.floor_line.count(Tile.BLUE) == 2
    assert player.pattern_lines[0] == []


def test_make_move_advances_current_player():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.make_move(Move(source=0, color=Tile.BLUE, destination=0))
    assert game.state.current_player == 1


def test_setup_round_places_first_player_token_in_center():
    game = Game()
    game.setup_round()
    assert Tile.FIRST_PLAYER in game.state.center


def test_legal_moves_allow_color_when_different_color_already_on_wall_row():
    # A different color being on the wall in row 0 should NOT block
    # other colors from being placed on pattern line 0.
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    # Fill factories with a known color so we control what's available
    for factory in game.state.factories:
        factory.clear()
        factory.extend([Tile.BLUE] * TILES_PER_FACTORY)
    # Place Yellow on row 0's wall (Yellow belongs in col 1, not col 0)
    player.wall[0][1] = Tile.YELLOW
    # Blue should still be placeable on pattern line 0
    moves = game.legal_moves()
    blue_to_row0 = [m for m in moves if m.destination == 0 and m.color == Tile.BLUE]
    assert len(blue_to_row0) > 0


def test_score_floor_does_not_send_first_player_tile_to_discard():
    game = Game()
    player = game.state.players[0]
    player.floor_line = [Tile.FIRST_PLAYER, Tile.BLUE]
    game._score_floor(player)
    assert Tile.FIRST_PLAYER not in game.state.discard
    assert Tile.BLUE in game.state.discard
