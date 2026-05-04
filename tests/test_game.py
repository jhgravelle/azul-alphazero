# tests/test_game.py
"""Tests for core game methods."""

from engine.game import CENTER, FLOOR, Game, Move
from engine.constants import (
    BONUS_ROW,
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
    assert Tile.FIRST_PLAYER in game.center


def test_setup_round_fills_factories():
    game = Game()
    game.setup_round()
    for factory in game.factories:
        assert len(factory) == TILES_PER_FACTORY


def test_setup_round_draws_from_bag():
    game = Game()
    initial_bag_size = len(game.bag)
    game.setup_round()
    num_factories = len(game.factories)
    expected_tiles_drawn = num_factories * TILES_PER_FACTORY
    assert len(game.bag) == initial_bag_size - expected_tiles_drawn


def test_setup_round_uses_discard_when_bag_empty():
    game = Game()
    game.discard = game.bag.copy()
    game.bag.clear()
    game.setup_round()
    for factory in game.factories:
        assert len(factory) == TILES_PER_FACTORY


def test_setup_round_partial_fill_when_no_tiles():
    game = Game()
    game.bag.clear()
    game.discard.clear()
    game.setup_round()
    total_tiles = sum(len(f) for f in game.factories)
    assert total_tiles == 0


def test_setup_round_fills_factories_in_order():
    game = Game()
    game.bag = [Tile.BLUE] * (TILES_PER_FACTORY + 1)
    game.discard.clear()
    game.setup_round()
    assert len(game.factories[0]) == TILES_PER_FACTORY
    assert len(game.factories[1]) == 1
    for factory in game.factories[2:]:
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
            source_tiles = game.center
        else:
            source_tiles = game.factories[move.source]
        assert move.tile in source_tiles


def test_legal_moves_empty_center_generates_no_center_moves():
    game = Game()
    game.setup_round()
    game.center.clear()
    center_moves = [m for m in game.legal_moves() if m.source == CENTER]
    assert len(center_moves) == 0


def test_legal_moves_exclude_full_pattern_line():
    game = Game()
    game.setup_round()
    game.current_player.pattern_lines[0] = [Tile.BLUE]
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0]


def test_legal_moves_exclude_pattern_line_with_different_tile():
    game = Game()
    game.setup_round()
    game.current_player.pattern_lines[0].append(Tile.BLUE)
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0 and m.tile != Tile.BLUE]


def test_legal_moves_exclude_pattern_line_where_wall_row_has_tile():
    game = Game()
    game.setup_round()
    game.current_player.wall[0][0] = Tile.BLUE
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0 and m.tile == Tile.BLUE]


def test_legal_moves_always_include_floor_move_for_every_available_tile():
    game = Game()
    for factory in game.factories:
        factory.clear()
    game.factories[0] = [Tile.BLUE] * TILES_PER_FACTORY
    game.current_player.pattern_lines[0] = [Tile.BLUE]
    for row in range(1, BOARD_SIZE):
        game.current_player.wall[row][
            COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][row]
        ] = Tile.BLUE
    floor_moves = [
        m for m in game.legal_moves() if m.destination == FLOOR and m.tile == Tile.BLUE
    ]
    assert len(floor_moves) == 1


def test_legal_moves_allow_tile_when_different_tile_already_on_wall_row():
    game = Game()
    game.setup_round()
    for factory in game.factories:
        factory.clear()
        factory.extend([Tile.BLUE] * TILES_PER_FACTORY)
    game.current_player.wall[0][1] = Tile.YELLOW
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
    """Return a game with one factory loaded and the rest empty."""
    game = Game()
    game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    return game


def test_make_move_removes_tile_from_factory():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.factories[0]


def test_make_move_leftover_factory_tiles_go_to_center():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.RED in game.center
    assert Tile.YELLOW in game.center


def test_make_move_taking_from_center_leaves_no_leftover():
    game = Game()
    game.center = [Tile.BLUE, Tile.BLUE, Tile.RED]
    game.factories[0] = [Tile.YELLOW] * TILES_PER_FACTORY
    game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.center
    assert Tile.RED in game.center


def test_make_move_taking_from_center_moves_first_player_marker_to_floor():
    game = Game()
    game.center = [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE, Tile.RED]
    game.factories[0] = [Tile.YELLOW] * TILES_PER_FACTORY
    game.make_move(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.FIRST_PLAYER in game.players[0].floor_line
    assert Tile.FIRST_PLAYER not in game.center


def test_make_move_places_tiles_on_pattern_line():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=1))
    assert game.players[0].pattern_lines[1].count(Tile.BLUE) == 2


def test_make_move_overflow_tiles_go_to_floor():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    player = game.players[0]
    assert player.pattern_lines[0] == [Tile.BLUE]
    assert player.floor_line.count(Tile.BLUE) == 1


def test_make_move_to_floor_puts_all_tiles_on_floor():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=FLOOR))
    player = game.players[0]
    assert player.floor_line.count(Tile.BLUE) == 2
    assert player.pattern_lines[0] == []


# endregion


# region score_round -------------------------------------------------------


def test_full_pattern_line_moves_tile_to_wall():
    game = Game()
    player = game.players[0]
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.wall[0][0] == Tile.BLUE


def test_completed_line_remaining_tiles_go_to_discard():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.YELLOW, Tile.YELLOW])
    game._score_round()
    assert game.discard.count(Tile.YELLOW) == 1


def test_incomplete_pattern_line_is_unchanged():
    game = Game()
    player = game.players[0]
    player.place(2, [Tile.RED])
    game._score_round()
    assert player.pattern_lines[2] == [Tile.RED]
    assert player.wall[2][2] is None


def test_tile_with_no_neighbours_scores_one_point():
    game = Game()
    player = game.players[0]
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.score == 1


def test_tile_with_horizontal_neighbours_scores_run_length():
    game = Game()
    player = game.players[0]
    player.wall[0][1] = Tile.YELLOW
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.score == 2


def test_tile_with_vertical_neighbours_scores_run_length():
    game = Game()
    player = game.players[0]
    player.wall[1][0] = Tile.WHITE
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.score == 2


def test_tile_with_both_neighbours_scores_combined_run_lengths():
    game = Game()
    player = game.players[0]
    player.wall[0][1] = Tile.YELLOW
    player.wall[1][0] = Tile.WHITE
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.score == 4


def test_floor_penalties_are_applied():
    game = Game()
    player = game.players[0]
    player.score = 10
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -1, -1, -2 = -4
    player._update_penalty()
    game._score_round()
    assert player.score == 6


def test_score_does_not_go_below_zero():
    game = Game()
    player = game.players[0]
    player.score = 1
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]
    player._update_penalty()
    game._score_round()
    assert player.score == 0


def test_floor_line_is_cleared_after_scoring():
    game = Game()
    game.players[0].floor_line = [Tile.BLUE, Tile.RED]
    game.players[0]._update_penalty()
    game._score_round()
    assert game.players[0].floor_line == []


def test_floor_tiles_except_first_player_go_to_discard():
    game = Game()
    player = game.players[0]
    player.floor_line = [Tile.FIRST_PLAYER, Tile.BLUE]
    player._update_penalty()
    game._score_round()
    assert Tile.FIRST_PLAYER not in game.discard
    assert Tile.BLUE in game.discard


def test_player_with_first_player_tile_starts_next_round():
    game = Game()
    game.players[1].floor_line = [Tile.FIRST_PLAYER]
    game.players[1]._update_penalty()
    game._score_round()
    assert game.current_player_index == 1


# endregion


# region is_game_over ------------------------------------------------------


def test_game_is_not_over_with_empty_walls():
    assert Game().is_game_over() is False


def test_game_is_not_over_with_incomplete_row():
    game = Game()
    game.players[0].wall[0] = [Tile.BLUE, Tile.YELLOW, None, None, None]
    assert game.is_game_over() is False


def test_game_is_over_when_one_player_completes_a_row():
    game = Game()
    game.players[0].wall[0] = [
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
        Tile.BLACK,
        Tile.WHITE,
    ]
    assert game.is_game_over() is True


def test_game_is_over_when_second_player_completes_a_row():
    game = Game()
    game.players[1].wall[2] = [
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
    game.players[0].wall[0] = [
        Tile.BLUE,
        Tile.YELLOW,
        Tile.RED,
        Tile.BLACK,
        Tile.WHITE,
    ]
    game.players[0]._update_bonus()
    game._score_game()
    assert game.players[0].score == 2


def test_two_complete_rows_scores_four_points():
    game = Game()
    p = game.players[0]
    p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    p.wall[1] = [Tile.WHITE, Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK]
    p._update_bonus()
    game._score_game()
    assert p.score == 4


def test_complete_column_scores_seven_points():
    game = Game()
    p = game.players[0]
    for row in range(BOARD_SIZE):
        p.wall[row][0] = WALL_PATTERN[row][0]
    p._update_bonus()
    game._score_game()
    assert p.score == 7


def test_complete_tile_color_scores_ten_points():
    game = Game()
    p = game.players[0]
    for row in range(BOARD_SIZE):
        p.wall[row][COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][row]] = Tile.BLUE
    p._update_bonus()
    game._score_game()
    assert p.score == 10


def test_score_game_combines_all_bonuses():
    game = Game()
    p = game.players[0]
    p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
    for row in range(BOARD_SIZE):
        p.wall[row][0] = WALL_PATTERN[row][0]
    for row in range(BOARD_SIZE):
        p.wall[row][COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][row]] = Tile.BLUE
    p._update_bonus()
    game._score_game()
    assert p.score == 2 + 7 + 10


def test_score_game_applies_to_all_players():
    game = Game()
    for p in game.players:
        p.wall[0] = [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]
        p._update_bonus()
    game._score_game()
    for p in game.players:
        assert p.score == 2


def test_score_game_bonuses_applied_on_game_over():
    game = Game()
    game.setup_round()
    player = game.players[0]

    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]

    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.factories[0] = [Tile.BLUE]
    game.current_player_index = 0

    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    game.advance()

    assert game.is_game_over()
    assert player.score >= BONUS_ROW


def test_score_game_not_applied_mid_game():
    game = Game()
    game.setup_round()
    player = game.players[0]

    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]

    assert player.score == 0


# endregion


# region clone -------------------------------------------------------------


def test_clone_returns_different_object():
    game = Game()
    game.setup_round()
    assert game.clone() is not game


def test_clone_state_is_equal():
    game = Game()
    game.setup_round()
    clone = game.clone()
    assert clone.current_player_index == game.current_player_index
    assert clone.round == game.round
    assert clone.center == game.center
    assert clone.bag == game.bag
    assert clone.discard == game.discard


def test_clone_factories_equal_but_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    assert clone.factories == game.factories
    clone.factories[0].clear()
    assert game.factories[0] != clone.factories[0]


def test_clone_center_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.center.append(Tile.BLUE)
    assert Tile.BLUE not in game.center


def test_clone_bag_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    original_len = len(game.bag)
    clone.bag.pop()
    assert len(game.bag) == original_len


def test_clone_discard_independent():
    game = Game()
    game.setup_round()
    game.discard = [Tile.RED, Tile.BLUE]
    clone = game.clone()
    clone.discard.append(Tile.BLACK)
    assert len(game.discard) == 2


def test_clone_player_score_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.players[0].score = 99
    assert game.players[0].score != 99


def test_clone_player_wall_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.players[0].wall[0][0] = Tile.BLUE
    assert game.players[0].wall[0][0] is None


def test_clone_player_pattern_lines_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.players[0].pattern_lines[0].append(Tile.BLUE)
    assert game.players[0].pattern_lines[0] == []


def test_clone_player_floor_line_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.players[0].floor_line.append(Tile.RED)
    assert game.players[0].floor_line == []


def test_clone_make_move_does_not_affect_original():
    game = Game()
    game.setup_round()
    clone = game.clone()
    move = clone.legal_moves()[0]
    clone.make_move(move)
    assert game.current_player_index == 0
    assert game.factories == game.clone().factories


# endregion


# region advance -----------------------------------------------------------


def test_advance_sets_up_next_round_when_round_ends():
    game = Game()
    game.setup_round()
    while not game.is_round_over():
        game.make_move(game.legal_moves()[0])
    game.advance()
    total_tiles = sum(len(f) for f in game.factories)
    assert total_tiles > 0


def test_advance_rotates_player():
    game = _mid_round_game()
    game.factories[1] = [Tile.RED] * TILES_PER_FACTORY
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=1))
    game.advance()
    assert game.current_player_index == 1


def test_advance_scores_round_when_round_ends():
    game = Game()
    game.setup_round()
    player = game.players[0]
    player.pattern_lines[0] = [Tile.BLUE]
    player._update_pending()
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.advance()
    assert player.wall[0][0] == Tile.BLUE


def test_advance_skips_setup_when_skip_setup_true():
    game = Game()
    game.setup_round()
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    assert game.advance(skip_setup=True) is True
    total = sum(len(f) for f in game.factories)
    assert total == 0


def test_advance_calls_setup_round_when_skip_setup_false():
    game = Game()
    game.setup_round()
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    assert game.advance(skip_setup=False) is True
    total = sum(len(f) for f in game.factories)
    assert total > 0


def test_advance_does_not_call_on_round_setup_when_game_ends():
    game = Game()
    game.setup_round()
    player = game.players[0]
    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.factories[0] = [Tile.BLUE]
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    assert game.advance(skip_setup=True) is True
    assert game.is_game_over()


def test_advance_scores_game_when_game_ends():
    game = Game()
    game.setup_round()
    player = game.players[0]
    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.factories[0] = [Tile.BLUE]
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    game.advance()
    assert game.is_game_over()
    assert player.score >= BONUS_ROW


def test_advance_does_nothing_mid_round():
    game = _mid_round_game()
    game.factories[1] = [Tile.RED] * TILES_PER_FACTORY
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=1))
    round_before = game.round
    result = game.advance()
    assert result is False
    assert game.round == round_before


# endregion


# region setup_round with explicit factories -------------------------------


def test_setup_round_with_explicit_factories():
    game = Game()
    explicit = [
        [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW],
        [Tile.BLACK, Tile.BLACK, Tile.WHITE, Tile.WHITE],
        [Tile.RED, Tile.RED, Tile.RED, Tile.RED],
        [Tile.YELLOW, Tile.YELLOW, Tile.BLUE, Tile.BLACK],
        [Tile.WHITE, Tile.RED, Tile.YELLOW, Tile.BLACK],
    ]
    bag_before = len(game.bag)
    game.setup_round(factories=explicit)
    assert game.factories == explicit
    assert len(game.bag) == bag_before


def test_setup_round_with_none_draws_randomly():
    game = Game()
    bag_before = len(game.bag)
    game.setup_round()
    total_drawn = sum(len(f) for f in game.factories)
    assert len(game.bag) == bag_before - total_drawn


def test_setup_round_increments_round_number():
    game = Game()
    assert game.round == 0
    game.setup_round()
    assert game.round == 1
    game.setup_round()
    assert game.round == 2


def test_setup_round_places_first_player_marker_in_center():
    game = Game()
    game.setup_round()
    assert Tile.FIRST_PLAYER in game.center


# endregion


# region _take_from_source -------------------------------------------------


def test_take_from_source_returns_chosen_tiles():
    game = Game()
    game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    chosen = game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert chosen.count(Tile.BLUE) == 2


def test_take_from_source_removes_chosen_from_factory():
    game = Game()
    game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.factories[0]


def test_take_from_source_sends_leftovers_to_center():
    game = Game()
    game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.RED in game.center
    assert Tile.YELLOW in game.center


def test_take_from_source_clears_factory():
    game = Game()
    game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert game.factories[0] == []


def test_take_from_source_from_center_leaves_no_leftover():
    game = Game()
    game.center = [Tile.BLUE, Tile.BLUE, Tile.RED]
    game._take_from_source(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.center
    assert Tile.RED in game.center


def test_take_from_source_includes_first_player_when_present():
    game = Game()
    game.center = [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE]
    chosen = game._take_from_source(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.FIRST_PLAYER in chosen


def test_take_from_source_removes_first_player_from_center():
    game = Game()
    game.center = [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE]
    game._take_from_source(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.FIRST_PLAYER not in game.center


def test_take_from_source_does_not_send_first_player_to_center():
    game = Game()
    game.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.center = [Tile.FIRST_PLAYER]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert game.center.count(Tile.FIRST_PLAYER) == 1


# endregion


# region player.place (replaces _place_tiles) ------------------------------


def test_place_puts_chosen_on_pattern_line():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.BLUE, Tile.BLUE])
    assert player.pattern_lines[1].count(Tile.BLUE) == 2


def test_place_overflow_goes_to_floor():
    game = Game()
    player = game.players[0]
    player.place(0, [Tile.BLUE, Tile.BLUE])
    assert player.pattern_lines[0] == [Tile.BLUE]
    assert player.floor_line.count(Tile.BLUE) == 1


def test_place_to_floor_destination_puts_all_on_floor():
    game = Game()
    player = game.players[0]
    player.place(FLOOR, [Tile.BLUE, Tile.BLUE])
    assert player.floor_line.count(Tile.BLUE) == 2
    assert player.pattern_lines[0] == []


def test_place_puts_first_player_on_floor():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.BLUE, Tile.BLUE, Tile.FIRST_PLAYER])
    assert Tile.FIRST_PLAYER in player.floor_line


def test_place_first_player_does_not_go_on_pattern_line():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.BLUE, Tile.BLUE, Tile.FIRST_PLAYER])
    assert Tile.FIRST_PLAYER not in player.pattern_lines[1]


# endregion


# region is_round_over -----------------------------------------------------


def test_is_round_over_false_when_factories_have_tiles():
    game = Game()
    game.setup_round()
    assert game.is_round_over() is False


def test_is_round_over_true_when_all_sources_empty():
    game = Game()
    game.setup_round()
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    assert game.is_round_over() is True


def test_is_round_over_true_when_center_has_only_first_player():
    game = Game()
    game.setup_round()
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.center.append(Tile.FIRST_PLAYER)
    assert game.is_round_over() is True


def test_is_round_over_false_when_center_has_color_tile():
    game = Game()
    game.setup_round()
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.center.append(Tile.FIRST_PLAYER)
    game.center.append(Tile.BLUE)
    assert game.is_round_over() is False


# endregion


# region seeded RNG --------------------------------------------------------


def test_game_with_same_seed_produces_identical_factories():
    game_a = Game(seed=42)
    game_a.setup_round()
    factories_a = [list(f) for f in game_a.factories]

    game_b = Game(seed=42)
    game_b.setup_round()
    factories_b = [list(f) for f in game_b.factories]

    assert factories_a == factories_b


def test_game_with_different_seeds_likely_produces_different_factories():
    game_a = Game(seed=1)
    game_a.setup_round()
    factories_a = [list(f) for f in game_a.factories]

    game_b = Game(seed=2)
    game_b.setup_round()
    factories_b = [list(f) for f in game_b.factories]

    assert factories_a != factories_b


def test_game_seed_covers_multiple_rounds():
    def play_one_round(seed: int):
        game = Game(seed=seed)
        game.setup_round()
        factories_round_1 = [list(f) for f in game.factories]
        while not game.is_round_over():
            move = game.legal_moves()[0]
            game.make_move(move)
            game.advance(skip_setup=True)
        game.setup_round()
        factories_round_2 = [list(f) for f in game.factories]
        return factories_round_1, factories_round_2

    factories_1a, factories_2a = play_one_round(99)
    factories_1b, factories_2b = play_one_round(99)

    assert factories_1a == factories_1b
    assert factories_2a == factories_2b


def test_game_without_seed_is_not_deterministic():
    game_a = Game()
    game_a.setup_round()
    factories_a = [list(f) for f in game_a.factories]

    game_b = Game()
    game_b.setup_round()
    factories_b = [list(f) for f in game_b.factories]

    different = factories_a != factories_b
    if not different:
        for _ in range(4):
            g = Game()
            g.setup_round()
            if [list(f) for f in g.factories] != factories_a:
                different = True
                break
    assert different


# endregion
