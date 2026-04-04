# tests/test_game.py

"""Tests for core game methods."""

from engine.game import CENTER, FLOOR, Game, Move
from engine.constants import (
    Tile,
    BOARD_SIZE,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
    TILES_PER_FACTORY,
    WALL_PATTERN,
)

# region setup_round -------------------------------------------------------


def test_setup_round_places_first_player_token_in_center():
    game = Game()
    game.setup_round()
    assert Tile.FIRST_PLAYER in game.state.center


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


# endregion


# region legal_moves -------------------------------------------------------


def test_legal_moves_returns_moves_after_setup():
    game = Game()
    game.setup_round()
    moves = game.legal_moves()
    assert len(moves) > 0


def test_legal_moves_only_contain_tiles_present_in_source():
    game = Game()
    game.setup_round()
    for move in game.legal_moves():
        if move.source == CENTER:
            source_tiles = game.state.center
        else:
            source_tiles = game.state.factories[move.source]
        assert move.tile in source_tiles


def test_legal_moves_empty_center_generates_no_center_moves():
    game = Game()
    game.setup_round()
    game.state.center.clear()
    center_moves = [m for m in game.legal_moves() if m.source == CENTER]
    assert len(center_moves) == 0


def test_legal_moves_exclude_full_pattern_line():
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    player.pattern_lines[0] = [Tile.BLUE]  # row 0 holds max 1 tile
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0]


def test_legal_moves_exclude_pattern_line_with_different_tile():
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    player.pattern_lines[0].append(Tile.BLUE)
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0 and m.tile != Tile.BLUE]


def test_legal_moves_exclude_pattern_line_where_wall_row_has_tile():
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    player.wall[0][0] = Tile.BLUE
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0 and m.tile == Tile.BLUE]


def test_legal_moves_always_include_floor_move_for_every_available_tile():
    # Block all pattern lines so FLOOR is the only valid destination.
    # This confirms _is_valid_destination always returns True for FLOOR.
    game = Game()
    for factory in game.state.factories:
        factory.clear()
    game.state.factories[0] = [Tile.BLUE] * TILES_PER_FACTORY
    player = game.state.players[game.state.current_player]
    player.pattern_lines[0] = [Tile.BLUE]  # full (capacity 1)
    for row in range(1, BOARD_SIZE):
        player.wall[row][COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][row]] = Tile.BLUE
    floor_moves = [
        m for m in game.legal_moves() if m.destination == FLOOR and m.tile == Tile.BLUE
    ]
    assert len(floor_moves) == 1


def test_legal_moves_allow_tile_when_different_tile_already_on_wall_row():
    game = Game()
    game.setup_round()
    player = game.state.players[game.state.current_player]
    for factory in game.state.factories:
        factory.clear()
        factory.extend([Tile.BLUE] * TILES_PER_FACTORY)
    player.wall[0][1] = Tile.YELLOW  # Yellow in col 1 — should not block Blue
    moves = game.legal_moves()
    assert [m for m in moves if m.destination == 0 and m.tile == Tile.BLUE]


# endregion


# region WALL_PATTERN / COLUMN_FOR_TILE_IN_ROW ----------------------------


def test_wall_pattern_is_correct_size():
    assert len(WALL_PATTERN) == BOARD_SIZE
    for row in WALL_PATTERN:
        assert len(row) == BOARD_SIZE


def test_wall_pattern_each_row_and_column_has_all_tiles():
    tiles = set(t for t in Tile if t != Tile.FIRST_PLAYER)
    for i in range(BOARD_SIZE):
        assert set(WALL_PATTERN[i]) == tiles, f"Row {i} missing tiles"
        column = {WALL_PATTERN[r][i] for r in range(BOARD_SIZE)}
        assert column == tiles, f"Column {i} missing tiles"


def test_wall_pattern_known_positions():
    assert WALL_PATTERN[0][0] == Tile.BLUE
    assert WALL_PATTERN[0][4] == Tile.WHITE
    assert WALL_PATTERN[1][0] == Tile.WHITE
    assert WALL_PATTERN[1][1] == Tile.BLUE
    assert WALL_PATTERN[4][0] == Tile.YELLOW
    assert WALL_PATTERN[4][4] == Tile.BLUE


def test_column_for_tile_in_row_known_positions():
    assert COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0] == 0
    assert COLUMN_FOR_TILE_IN_ROW[Tile.WHITE][0] == 4
    assert COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][1] == 1
    assert COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][4] == 4


def test_column_for_tile_in_row_every_row_has_unique_columns():
    for row in range(BOARD_SIZE):
        columns = [COLUMN_FOR_TILE_IN_ROW[tile][row] for tile in COLOR_TILES]
        assert sorted(columns) == list(
            range(BOARD_SIZE)
        ), f"Row {row} columns not unique"


# endregion


# region make_move ---------------------------------------------------------


def _mid_round_game() -> Game:
    """Return a game with one factory loaded and the rest empty.

    The round will NOT end after a single make_move call, so player-advance
    and tile-placement assertions stay clean.
    """
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    return game


def test_make_move_removes_tile_from_factory():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.state.factories[0]


def test_make_move_leftover_factory_tiles_go_to_center():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.RED in game.state.center
    assert Tile.YELLOW in game.state.center


def test_make_move_taking_from_center_leaves_no_leftover():
    game = Game()
    game.state.center = [Tile.BLUE, Tile.BLUE, Tile.RED]
    game.state.factories[0] = [Tile.YELLOW] * TILES_PER_FACTORY  # keep round alive
    game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.state.center
    assert Tile.RED in game.state.center


def test_make_move_taking_from_center_moves_first_player_marker_to_floor():
    game = Game()
    game.state.center = [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE, Tile.RED]
    game.state.factories[0] = [Tile.YELLOW] * TILES_PER_FACTORY  # keep round alive
    game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    player = game.state.players[0]
    assert Tile.FIRST_PLAYER in player.floor_line
    assert Tile.FIRST_PLAYER not in game.state.center


def test_make_move_places_tiles_on_pattern_line():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=1))
    assert game.state.players[0].pattern_lines[1].count(Tile.BLUE) == 2


def test_make_move_overflow_tiles_go_to_floor():
    game = _mid_round_game()
    # Row 0 holds max 1 tile — 2 blues means 1 overflows
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    player = game.state.players[0]
    assert player.pattern_lines[0] == [Tile.BLUE]
    assert player.floor_line.count(Tile.BLUE) == 1


def test_make_move_to_floor_puts_all_tiles_on_floor():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=FLOOR))
    player = game.state.players[0]
    assert player.floor_line.count(Tile.BLUE) == 2
    assert player.pattern_lines[0] == []


def test_make_move_advances_current_player():
    game = _mid_round_game()
    game.state.factories[1] = [Tile.RED] * TILES_PER_FACTORY  # ensure round continues
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=1))
    assert game.state.current_player == 1


# endregion


# region score_round -------------------------------------------------------


def test_full_pattern_line_moves_tile_to_wall():
    game = Game()
    game.state.players[0].pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert game.state.players[0].wall[0][0] == Tile.BLUE


def test_completed_line_remaining_tiles_go_to_discard():
    game = Game()
    game.state.players[0].pattern_lines[1] = [Tile.YELLOW, Tile.YELLOW]
    game.score_round()
    assert game.state.discard.count(Tile.YELLOW) == 1


def test_incomplete_pattern_line_is_unchanged():
    game = Game()
    game.state.players[0].pattern_lines[2] = [Tile.RED]
    game.score_round()
    assert game.state.players[0].pattern_lines[2] == [Tile.RED]
    assert game.state.players[0].wall[2][2] is None


def test_tile_with_no_neighbours_scores_one_point():
    game = Game()
    game.state.players[0].pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert game.state.players[0].score == 1


def test_tile_with_horizontal_neighbours_scores_run_length():
    game = Game()
    player = game.state.players[0]
    player.wall[0][1] = Tile.YELLOW
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 2


def test_tile_with_vertical_neighbours_scores_run_length():
    game = Game()
    player = game.state.players[0]
    player.wall[1][0] = Tile.WHITE
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 2


def test_tile_with_both_neighbours_scores_combined_run_lengths():
    game = Game()
    player = game.state.players[0]
    player.wall[0][1] = Tile.YELLOW
    player.wall[1][0] = Tile.WHITE
    player.pattern_lines[0] = [Tile.BLUE]
    game.score_round()
    assert player.score == 4


def test_floor_penalties_are_applied():
    game = Game()
    player = game.state.players[0]
    player.score = 10
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -1, -1, -2 = -4
    game.score_round()
    assert player.score == 6


def test_score_does_not_go_below_zero():
    game = Game()
    player = game.state.players[0]
    player.score = 1
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.score_round()
    assert player.score == 0


def test_floor_line_is_cleared_after_scoring():
    game = Game()
    game.state.players[0].floor_line = [Tile.BLUE, Tile.RED]
    game.score_round()
    assert game.state.players[0].floor_line == []


def test_score_floor_does_not_send_first_player_tile_to_discard():
    game = Game()
    player = game.state.players[0]
    player.floor_line = [Tile.FIRST_PLAYER, Tile.BLUE]
    game._score_floor(player)
    assert Tile.FIRST_PLAYER not in game.state.discard
    assert Tile.BLUE in game.state.discard


def test_player_with_first_player_tile_starts_next_round():
    game = Game()
    game.state.players[1].floor_line = [Tile.FIRST_PLAYER]
    game.score_round()
    assert game.state.current_player == 1


# endregion


# region is_game_over ------------------------------------------------------


def test_game_is_not_over_with_empty_walls():
    assert Game().is_game_over() is False


def test_game_is_not_over_with_incomplete_row():
    game = Game()
    game.state.players[0].wall[0] = [Tile.BLUE, Tile.YELLOW, None, None, None]
    assert game.is_game_over() is False


def test_game_is_over_when_one_player_completes_a_row():
    game = Game()
    game.state.players[0].wall[0] = [
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
        Tile.BLACK,
        Tile.WHITE,
    ]
    assert game.is_game_over() is True


def test_game_is_over_when_second_player_completes_a_row():
    game = Game()
    game.state.players[1].wall[2] = [
        Tile.BLACK,
        Tile.WHITE,
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
    ]
    assert game.is_game_over() is True


# endregion


# region score_game --------------------------------------------------------


def test_complete_row_scores_two_points():
    game = Game()
    game.state.players[0].wall[0] = [
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
        Tile.BLACK,
        Tile.WHITE,
    ]
    game.score_game()
    assert game.state.players[0].score == 2


def test_two_complete_rows_scores_four_points():
    game = Game()
    p = game.state.players[0]
    p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    p.wall[1] = [Tile.WHITE, Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK]
    game.score_game()
    assert p.score == 4


def test_complete_column_scores_seven_points():
    game = Game()
    p = game.state.players[0]
    for row in range(BOARD_SIZE):
        p.wall[row][0] = WALL_PATTERN[row][0]
    game.score_game()
    assert p.score == 7


def test_complete_tile_color_scores_ten_points():
    game = Game()
    p = game.state.players[0]
    for row in range(BOARD_SIZE):
        p.wall[row][COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][row]] = Tile.BLUE
    game.score_game()
    assert p.score == 10


def test_score_game_combines_all_bonuses():
    game = Game()
    p = game.state.players[0]
    p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    for row in range(BOARD_SIZE):
        p.wall[row][0] = WALL_PATTERN[row][0]
    for row in range(BOARD_SIZE):
        p.wall[row][COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][row]] = Tile.BLUE
    game.score_game()
    assert p.score == 2 + 7 + 10


def test_score_game_applies_to_all_players():
    game = Game()
    for p in game.state.players:
        p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    game.score_game()
    for p in game.state.players:
        assert p.score == 2


# endregion
