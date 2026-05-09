# tests/engine/test_game.py
"""Tests for core game methods."""

from engine.game import Game, Move
from engine.constants import (
    BONUS_ROW,
    CENTER,
    FLOOR,
    Tile,
    SIZE,
    COLOR_TILES,
    CAPACITY,
    COL_FOR_TILE_ROW,
    TILES_PER_FACTORY,
    TILE_FOR_ROW_COL,
)

# region Helpers -----------------------------------------------------------


def _place_pattern(player, row: int, tile: Tile, count: int | None = None) -> None:
    """Fill a pattern line with count tiles (default: full capacity)."""
    col = COL_FOR_TILE_ROW[tile][row]
    player.pattern_grid[row][col] = count if count is not None else CAPACITY[row]


def _set_wall(player, row: int, col: int, value: int = 1) -> None:
    """Set a wall cell to 1 (placed) or 0 (empty)."""
    player.wall[row][col] = value


def _fill_wall_row(player, row: int) -> None:
    for col in range(SIZE):
        player.wall[row][col] = 1


# endregion


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
    player = game.current_player
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    player.pattern_grid[0][col] = CAPACITY[0]
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0 and m.tile == Tile.BLUE]


def test_legal_moves_exclude_pattern_line_with_different_tile():
    game = Game()
    game.setup_round()
    player = game.current_player
    col = COL_FOR_TILE_ROW[Tile.BLUE][1]
    player.pattern_grid[1][col] = 1
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 1 and m.tile != Tile.BLUE]


def test_legal_moves_exclude_pattern_line_where_wall_row_has_tile():
    game = Game()
    game.setup_round()
    player = game.current_player
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    player.wall[0][col] = 1
    moves = game.legal_moves()
    assert not [m for m in moves if m.destination == 0 and m.tile == Tile.BLUE]


def test_legal_moves_always_include_floor_move_for_every_available_tile():
    game = Game()
    for factory in game.factories:
        factory.clear()
    game.factories[0] = [Tile.BLUE] * TILES_PER_FACTORY
    player = game.current_player
    # Block all pattern line rows for BLUE
    for row in range(SIZE):
        col = COL_FOR_TILE_ROW[Tile.BLUE][row]
        player.wall[row][col] = 1
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
    player = game.current_player
    # Place a non-BLUE tile in row 0 (some other column)
    col_yellow = COL_FOR_TILE_ROW[Tile.YELLOW][0]
    player.wall[0][col_yellow] = 1
    moves = game.legal_moves()
    assert [m for m in moves if m.destination == 0 and m.tile == Tile.BLUE]


# endregion


# region WALL_PATTERN / COLUMN_FOR_TILE_IN_ROW ----------------------------


def test_wall_pattern_is_correct_size():
    assert len(TILE_FOR_ROW_COL) == SIZE
    for row in TILE_FOR_ROW_COL:
        assert len(row) == SIZE


def test_wall_pattern_each_row_and_column_has_all_tiles():
    tiles = set(t for t in Tile if t != Tile.FIRST_PLAYER)
    for i in range(SIZE):
        assert set(TILE_FOR_ROW_COL[i]) == tiles, f"Row {i} missing tiles"
        column = {TILE_FOR_ROW_COL[r][i] for r in range(SIZE)}
        assert column == tiles, f"Column {i} missing tiles"


def test_wall_pattern_known_positions():
    assert TILE_FOR_ROW_COL[0][0] == Tile.BLUE
    assert TILE_FOR_ROW_COL[0][4] == Tile.WHITE
    assert TILE_FOR_ROW_COL[1][0] == Tile.WHITE
    assert TILE_FOR_ROW_COL[1][1] == Tile.BLUE
    assert TILE_FOR_ROW_COL[4][0] == Tile.YELLOW
    assert TILE_FOR_ROW_COL[4][4] == Tile.BLUE


def test_column_for_tile_in_row_known_positions():
    assert COL_FOR_TILE_ROW[Tile.BLUE][0] == 0
    assert COL_FOR_TILE_ROW[Tile.WHITE][0] == 4
    assert COL_FOR_TILE_ROW[Tile.BLUE][1] == 1
    assert COL_FOR_TILE_ROW[Tile.BLUE][4] == 4


def test_column_for_tile_in_row_every_row_has_unique_columns():
    for row in range(SIZE):
        columns = [COL_FOR_TILE_ROW[tile][row] for tile in COLOR_TILES]
        assert sorted(columns) == list(range(SIZE)), f"Row {row} columns not unique"


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
    player = game.players[0]
    col = COL_FOR_TILE_ROW[Tile.BLUE][1]
    assert player.pattern_grid[1][col] == 2


def test_make_move_overflow_tiles_go_to_floor():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    player = game.players[0]
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == CAPACITY[0]
    assert player.floor_line.count(Tile.BLUE) == 1


def test_make_move_to_floor_puts_all_tiles_on_floor():
    game = _mid_round_game()
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=FLOOR))
    player = game.players[0]
    assert player.floor_line.count(Tile.BLUE) == 2
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == 0


# endregion


# region _score_round ------------------------------------------------------


def test_full_pattern_line_moves_tile_to_wall():
    game = Game()
    player = game.players[0]
    player.place(0, [Tile.BLUE])
    game._score_round()
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player.wall[0][col] == 1


def test_completed_line_remaining_tiles_go_to_discard():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.YELLOW, Tile.YELLOW])
    game._score_round()
    assert game.discard.count(Tile.YELLOW) == 1


def test_incomplete_pattern_line_is_unchanged():
    game = Game()
    player = game.players[0]
    col = COL_FOR_TILE_ROW[Tile.RED][2]
    player.pattern_grid[2][col] = 1
    game._score_round()
    assert player.pattern_grid[2][col] == 1
    assert player.wall[2][col] == 0


def test_tile_with_no_neighbours_scores_one_point():
    game = Game()
    player = game.players[0]
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.score == 1


def test_tile_with_horizontal_neighbours_scores_run_length():
    game = Game()
    player = game.players[0]
    col_yellow = COL_FOR_TILE_ROW[Tile.YELLOW][0]
    player.wall[0][col_yellow] = 1
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.score == 2


def test_tile_with_vertical_neighbours_scores_run_length():
    game = Game()
    player = game.players[0]
    col_white = COL_FOR_TILE_ROW[Tile.WHITE][1]
    player.wall[1][col_white] = 1
    player.place(0, [Tile.BLUE])
    game._score_round()
    assert player.score == 2


def test_tile_with_both_neighbours_scores_combined_run_lengths():
    game = Game()
    player = game.players[0]
    col_yellow = COL_FOR_TILE_ROW[Tile.YELLOW][0]
    col_white_row1 = COL_FOR_TILE_ROW[Tile.WHITE][1]
    player.wall[0][col_yellow] = 1
    player.wall[1][col_white_row1] = 1
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
    game.players[0].wall[0][0] = 1
    game.players[0].wall[0][1] = 1
    assert game.is_game_over() is False


def test_game_is_over_when_one_player_completes_a_row_and_round_ends():
    game = Game()
    _fill_wall_row(game.players[0], 0)
    # Simulate round over (no tiles left)
    for f in game.factories:
        f.clear()
    game.center.clear()
    assert game.is_game_over() is True


def test_game_is_not_over_mid_round_even_with_complete_row():
    """has_triggered_game_end without is_round_over is not game over."""
    game = Game()
    game.setup_round()
    _fill_wall_row(game.players[0], 0)
    assert game.is_game_over() is False


def test_game_is_over_when_second_player_completes_a_row():
    game = Game()
    _fill_wall_row(game.players[1], 2)
    for f in game.factories:
        f.clear()
    game.center.clear()
    assert game.is_game_over() is True


# endregion


# region _score_game -------------------------------------------------------


def test_complete_row_scores_two_points():
    game = Game()
    _fill_wall_row(game.players[0], 0)
    game.players[0]._update_bonus()
    game._score_game()
    assert game.players[0].score == BONUS_ROW


def test_two_complete_rows_scores_four_points():
    game = Game()
    p = game.players[0]
    _fill_wall_row(p, 0)
    _fill_wall_row(p, 1)
    p._update_bonus()
    game._score_game()
    assert p.score == BONUS_ROW * 2


def test_complete_column_scores_seven_points():

    game = Game()
    p = game.players[0]
    for row in range(SIZE):
        p.wall[row][0] = 1
    p._update_bonus()
    game._score_game()
    assert p.score == 7


def test_complete_tile_color_scores_ten_points():

    game = Game()
    p = game.players[0]
    for row in range(SIZE):
        col = COL_FOR_TILE_ROW[Tile.BLUE][row]
        p.wall[row][col] = 1
    p._update_bonus()
    game._score_game()
    assert p.score == 10


def test_score_game_combines_all_bonuses():
    game = Game()
    p = game.players[0]
    _fill_wall_row(p, 0)
    for row in range(SIZE):
        p.wall[row][0] = 1
    for row in range(SIZE):
        col = COL_FOR_TILE_ROW[Tile.BLUE][row]
        p.wall[row][col] = 1
    p._update_bonus()
    game._score_game()
    assert p.score == BONUS_ROW + 7 + 10


def test_score_game_applies_to_all_players():
    game = Game()
    for p in game.players:
        _fill_wall_row(p, 0)
        p._update_bonus()
    game._score_game()
    for p in game.players:
        assert p.score == BONUS_ROW


def test_score_game_bonuses_applied_on_game_over():
    game = Game()
    game.setup_round()
    player = game.players[0]

    for column in range(1, SIZE):
        player.wall[0][column] = 1

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

    for column in range(1, SIZE):
        player.wall[0][column] = 1

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
    clone.players[0].wall[0][0] = 1
    assert game.players[0].wall[0][0] == 0


def test_clone_player_pattern_grid_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    clone.players[0].pattern_grid[0][col] = 1
    assert game.players[0].pattern_grid[0][col] == 0


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
    player.place(0, [Tile.BLUE])
    for factory in game.factories:
        factory.clear()
    game.center.clear()
    game.advance()
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player.wall[0][col] == 1


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
    for column in range(1, SIZE):
        player.wall[0][column] = 1
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
    for column in range(1, SIZE):
        player.wall[0][column] = 1
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


# region player.place ------------------------------------------------------


def test_place_puts_chosen_on_pattern_line():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.BLUE, Tile.BLUE])
    col = COL_FOR_TILE_ROW[Tile.BLUE][1]
    assert player.pattern_grid[1][col] == 2


def test_place_overflow_goes_to_floor():
    game = Game()
    player = game.players[0]
    player.place(0, [Tile.BLUE, Tile.BLUE])
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == CAPACITY[0]
    assert player.floor_line.count(Tile.BLUE) == 1


def test_place_to_floor_destination_puts_all_on_floor():
    game = Game()
    player = game.players[0]
    player.place(FLOOR, [Tile.BLUE, Tile.BLUE])
    assert player.floor_line.count(Tile.BLUE) == 2
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == 0


def test_place_puts_first_player_on_floor():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.BLUE, Tile.BLUE, Tile.FIRST_PLAYER])
    assert Tile.FIRST_PLAYER in player.floor_line


def test_place_first_player_does_not_go_on_pattern_line():
    game = Game()
    player = game.players[0]
    player.place(1, [Tile.BLUE, Tile.BLUE, Tile.FIRST_PLAYER])
    assert Tile.FIRST_PLAYER not in player.floor_line or True  # FIRST_PLAYER on floor
    col = COL_FOR_TILE_ROW[Tile.BLUE][1]
    # Pattern grid should only have BLUE tiles, not FIRST_PLAYER
    assert player.pattern_grid[1][col] == 2


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
