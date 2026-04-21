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


def test_score_game_bonuses_applied_on_game_over():
    """End-of-game wall bonuses are applied automatically when the game ends.

    Player 0 has 4 tiles already in wall row 0 and a full pattern line that
    will place the 5th tile, completing the row and triggering game over.
    After the move resolves, player 0's score must include the +BONUS_ROW
    bonus on top of the placement score.
    """
    game = Game()
    game.setup_round()
    player = game.state.players[0]

    # Fill wall row 0 columns 1-4 — placing BLUE at column 0 completes the row.
    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]

    # One BLUE tile in factory 0 — this is the only remaining source tile.
    # BLUE goes to column 0 of row 0 per WALL_PATTERN.
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.factories[0] = [Tile.BLUE]
    game.state.current_player = 0

    # Make the move — takes BLUE into pattern line 0 (capacity 1, so it's full).
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    game.advance()

    # score_round fires (sources empty), places BLUE on wall, then
    # is_game_over returns True (row 0 complete), then score_game fires.
    assert game.is_game_over()
    assert player.score >= BONUS_ROW


def test_score_game_not_applied_mid_game():
    """Wall bonuses are not applied while the game is still in progress."""
    game = Game()
    game.setup_round()
    player = game.state.players[0]

    # Give player 0 a nearly complete wall row but don't finish it.
    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]

    # Score at this point should be 0 — no bonuses yet.
    assert player.score == 0


# endregion


# ── Game.clone ─────────────────────────────────────────────────────────────


def test_clone_returns_different_object():
    game = Game()
    game.setup_round()
    assert game.clone() is not game


def test_clone_state_is_equal():
    game = Game()
    game.setup_round()
    clone = game.clone()
    assert clone.state.current_player == game.state.current_player
    assert clone.state.round == game.state.round
    assert clone.state.center == game.state.center
    assert clone.state.bag == game.state.bag
    assert clone.state.discard == game.state.discard


def test_clone_factories_equal_but_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    assert clone.state.factories == game.state.factories
    clone.state.factories[0].clear()
    assert game.state.factories[0] != clone.state.factories[0]


def test_clone_center_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.state.center.append(Tile.BLUE)
    assert Tile.BLUE not in game.state.center


def test_clone_bag_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    original_len = len(game.state.bag)
    clone.state.bag.pop()
    assert len(game.state.bag) == original_len


def test_clone_discard_independent():
    game = Game()
    game.setup_round()
    game.state.discard = [Tile.RED, Tile.BLUE]
    clone = game.clone()
    clone.state.discard.append(Tile.BLACK)
    assert len(game.state.discard) == 2


def test_clone_player_score_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.state.players[0].score = 99
    assert game.state.players[0].score != 99


def test_clone_player_wall_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.state.players[0].wall[0][0] = Tile.BLUE
    assert game.state.players[0].wall[0][0] is None


def test_clone_player_pattern_lines_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.state.players[0].pattern_lines[0].append(Tile.BLUE)
    assert game.state.players[0].pattern_lines[0] == []


def test_clone_player_floor_line_independent():
    game = Game()
    game.setup_round()
    clone = game.clone()
    clone.state.players[0].floor_line.append(Tile.RED)
    assert game.state.players[0].floor_line == []


def test_clone_make_move_does_not_affect_original():
    game = Game()
    game.setup_round()
    clone = game.clone()
    move = clone.legal_moves()[0]
    clone.make_move(move)
    # Original game state should be unchanged
    assert game.state.current_player == 0
    assert game.state.factories == game.clone().state.factories


# region advance -----------------------------------------------------------


def test_advance_sets_up_next_round_when_round_ends():
    """advance() must refill factories after the round ends (game not over)."""
    game = Game()
    game.setup_round()

    while not game.is_round_over():
        game.make_move(game.legal_moves()[0])

    game.advance()

    total_tiles = sum(len(f) for f in game.state.factories)
    assert total_tiles > 0


def test_advance_rotates_player():
    game = _mid_round_game()
    game.state.factories[1] = [Tile.RED] * TILES_PER_FACTORY
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=1))
    game.advance()
    assert game.state.current_player == 1


def test_advance_scores_round_when_round_ends():
    game = Game()
    game.setup_round()
    player = game.state.players[0]
    player.pattern_lines[0] = [Tile.BLUE]
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.advance()
    assert player.wall[0][0] == Tile.BLUE


def test_advance_skips_setup_when_skip_setup_true():
    game = Game()
    game.setup_round()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()

    assert game.advance(skip_setup=True) is True
    total = sum(len(f) for f in game.state.factories)
    assert total == 0


def test_advance_calls_setup_round_when_skip_setup_false():
    game = Game()
    game.setup_round()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()

    assert game.advance(skip_setup=False) is True
    total = sum(len(f) for f in game.state.factories)
    assert total > 0


def test_advance_does_not_call_on_round_setup_when_game_ends():
    game = Game()
    game.setup_round()
    player = game.state.players[0]
    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.factories[0] = [Tile.BLUE]
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))

    # skip_setup=True should have no effect when game is over
    assert game.advance(skip_setup=True) is True
    assert game.is_game_over()


def test_advance_scores_game_when_game_ends():
    game = Game()
    game.setup_round()
    player = game.state.players[0]
    for column in range(1, BOARD_SIZE):
        player.wall[0][column] = WALL_PATTERN[0][column]
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.factories[0] = [Tile.BLUE]
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=0))
    game.advance()
    assert game.is_game_over()
    assert player.score >= BONUS_ROW


def test_advance_does_nothing_mid_round():
    game = _mid_round_game()
    game.state.factories[1] = [Tile.RED] * TILES_PER_FACTORY
    game.make_move(Move(source=0, tile=Tile.BLUE, destination=1))
    round_before = game.state.round
    result = game.advance()
    assert result is False
    assert game.state.round == round_before


# endregion


def test_setup_round_with_explicit_factories():
    """setup_round should load the provided tile lists instead of drawing
    from the bag."""
    from engine.constants import Tile

    game = Game()
    explicit = [
        [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW],
        [Tile.BLACK, Tile.BLACK, Tile.WHITE, Tile.WHITE],
        [Tile.RED, Tile.RED, Tile.RED, Tile.RED],
        [Tile.YELLOW, Tile.YELLOW, Tile.BLUE, Tile.BLACK],
        [Tile.WHITE, Tile.RED, Tile.YELLOW, Tile.BLACK],
    ]
    bag_before = len(game.state.bag)
    game.setup_round(factories=explicit)

    assert game.state.factories == explicit
    # Bag must be untouched -- no random draw occurred.
    assert len(game.state.bag) == bag_before


def test_setup_round_with_none_draws_randomly():
    """setup_round() with no argument should draw from the bag as before."""
    game = Game()
    bag_before = len(game.state.bag)
    game.setup_round()
    total_drawn = sum(len(f) for f in game.state.factories)
    assert len(game.state.bag) == bag_before - total_drawn


def test_setup_round_increments_round_number():
    game = Game()
    assert game.state.round == 0
    game.setup_round()
    assert game.state.round == 1
    game.setup_round()
    assert game.state.round == 2


def test_setup_round_places_first_player_marker_in_center():
    game = Game()
    game.setup_round()
    assert Tile.FIRST_PLAYER in game.state.center


# endregion
# region clamped_points ----------------------------------------------------


def test_clamped_points_initialized_to_zero():
    game = Game()
    assert game.state.players[0].clamped_points == 0
    assert game.state.players[1].clamped_points == 0


def test_clamped_points_zero_when_floor_does_not_cause_clamp():
    game = Game()
    player = game.state.players[0]
    player.score = 10
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -4
    game._score_floor(player)
    assert player.score == 6
    assert player.clamped_points == 0


def test_clamped_points_records_amount_lost_to_clamp():
    game = Game()
    player = game.state.players[0]
    player.score = 1
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -4
    game._score_floor(player)
    assert player.score == 0  # clamped
    assert player.clamped_points == 3  # 3 points lost to clamp


def test_clamped_points_accumulates_across_rounds():
    game = Game()
    player = game.state.players[0]
    player.score = 1
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -4, clamps 3
    game._score_floor(player)
    assert player.clamped_points == 3

    player.score = 2
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -4, clamps 2
    game._score_floor(player)
    assert player.clamped_points == 5  # 3 + 2


def test_full_floor_line_clamped_points():
    game = Game()
    player = game.state.players[0]
    # Full floor: -1-1-2-2-2-3-3 = -14, score starts at 0
    player.floor_line = [
        Tile.BLUE,
        Tile.RED,
        Tile.YELLOW,
        Tile.BLACK,
        Tile.WHITE,
        Tile.BLUE,
        Tile.RED,
    ]
    game._score_floor(player)
    assert player.score == 0
    assert player.clamped_points == 14


def test_raw_score_is_score_plus_clamped_points():
    game = Game()
    player = game.state.players[0]
    player.score = 1
    player.floor_line = [Tile.BLUE, Tile.RED, Tile.YELLOW]  # -4
    game._score_floor(player)
    raw = player.score - player.clamped_points
    assert raw == -3  # true unclamped score: 1 - 4


def test_clone_preserves_clamped_points():
    game = Game()
    game.state.players[0].clamped_points = 7
    clone = game.clone()
    assert clone.state.players[0].clamped_points == 7


def test_clone_clamped_points_is_independent():
    game = Game()
    game.state.players[0].clamped_points = 7
    clone = game.clone()
    clone.state.players[0].clamped_points = 99
    assert game.state.players[0].clamped_points == 7


# endregion
# region is_round_over -----------------------------------------------------


def test_is_round_over_false_when_factories_have_tiles():
    game = Game()
    game.setup_round()
    assert game.is_round_over() is False


def test_is_round_over_true_when_all_sources_empty():
    game = Game()
    game.setup_round()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    assert game.is_round_over() is True


def test_is_round_over_true_when_center_has_only_first_player():
    game = Game()
    game.setup_round()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.center.append(Tile.FIRST_PLAYER)
    assert game.is_round_over() is True


def test_is_round_over_false_when_center_has_color_tile():
    game = Game()
    game.setup_round()
    for factory in game.state.factories:
        factory.clear()
    game.state.center.clear()
    game.state.center.append(Tile.FIRST_PLAYER)
    game.state.center.append(Tile.BLUE)
    assert game.is_round_over() is False


# endregion
# region _take_from_source -------------------------------------------------


def test_take_from_source_returns_chosen_tiles():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    chosen = game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert chosen.count(Tile.BLUE) == 2


def test_take_from_source_removes_chosen_from_factory():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.state.factories[0]


def test_take_from_source_sends_leftovers_to_center():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert Tile.RED in game.state.center
    assert Tile.YELLOW in game.state.center


def test_take_from_source_clears_factory():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert game.state.factories[0] == []


def test_take_from_source_from_center_leaves_no_leftover():
    game = Game()
    game.state.center = [Tile.BLUE, Tile.BLUE, Tile.RED]
    game._take_from_source(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.BLUE not in game.state.center
    assert Tile.RED in game.state.center


def test_take_from_source_includes_first_player_when_present():
    game = Game()
    game.state.center = [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE]
    chosen = game._take_from_source(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.FIRST_PLAYER in chosen


def test_take_from_source_removes_first_player_from_center():
    game = Game()
    game.state.center = [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE]
    game._take_from_source(Move(source=CENTER, tile=Tile.BLUE, destination=0))
    assert Tile.FIRST_PLAYER not in game.state.center


def test_take_from_source_does_not_send_first_player_to_center():
    game = Game()
    game.state.factories[0] = [Tile.BLUE, Tile.BLUE, Tile.RED, Tile.YELLOW]
    game.state.center = [Tile.FIRST_PLAYER]
    game._take_from_source(Move(source=0, tile=Tile.BLUE, destination=0))
    assert game.state.center.count(Tile.FIRST_PLAYER) == 1


# endregion


# region _place_tiles -------------------------------------------------------


def test_place_tiles_puts_chosen_on_pattern_line():
    game = Game()
    player = game.state.players[0]
    game._place_tiles(
        player, Move(source=0, tile=Tile.BLUE, destination=1), [Tile.BLUE, Tile.BLUE]
    )
    assert player.pattern_lines[1].count(Tile.BLUE) == 2


def test_place_tiles_overflow_goes_to_floor():
    game = Game()
    player = game.state.players[0]
    # Row 0 holds max 1 tile — 2 blues means 1 overflows
    game._place_tiles(
        player, Move(source=0, tile=Tile.BLUE, destination=0), [Tile.BLUE, Tile.BLUE]
    )
    assert player.pattern_lines[0] == [Tile.BLUE]
    assert player.floor_line.count(Tile.BLUE) == 1


def test_place_tiles_to_floor_destination_puts_all_on_floor():
    game = Game()
    player = game.state.players[0]
    game._place_tiles(
        player,
        Move(source=0, tile=Tile.BLUE, destination=FLOOR),
        [Tile.BLUE, Tile.BLUE],
    )
    assert player.floor_line.count(Tile.BLUE) == 2
    assert player.pattern_lines[0] == []


def test_place_tiles_puts_first_player_on_floor():
    game = Game()
    player = game.state.players[0]
    game._place_tiles(
        player,
        Move(source=CENTER, tile=Tile.BLUE, destination=1),
        [Tile.BLUE, Tile.BLUE, Tile.FIRST_PLAYER],
    )
    assert Tile.FIRST_PLAYER in player.floor_line


def test_place_tiles_first_player_does_not_go_on_pattern_line():
    game = Game()
    player = game.state.players[0]
    game._place_tiles(
        player,
        Move(source=CENTER, tile=Tile.BLUE, destination=1),
        [Tile.BLUE, Tile.BLUE, Tile.FIRST_PLAYER],
    )
    assert Tile.FIRST_PLAYER not in player.pattern_lines[1]


# endregion
# region GameState.clone ---------------------------------------------------


def test_game_state_clone_returns_different_object():
    game = Game()
    game.setup_round()
    assert game.state.clone() is not game.state


def test_game_state_clone_scalar_fields_match():
    game = Game()
    game.setup_round()
    game.state.current_player = 1
    game.state.round = 3
    clone = game.state.clone()
    assert clone.current_player == 1
    assert clone.round == 3


def test_game_state_clone_factories_equal_but_independent():
    game = Game()
    game.setup_round()
    clone = game.state.clone()
    assert clone.factories == game.state.factories
    clone.factories[0].clear()
    assert game.state.factories[0] != clone.factories[0]


def test_game_state_clone_center_independent():
    game = Game()
    game.setup_round()
    clone = game.state.clone()
    clone.center.append(Tile.BLUE)
    assert Tile.BLUE not in game.state.center


def test_game_state_clone_bag_independent():
    game = Game()
    game.setup_round()
    clone = game.state.clone()
    original_len = len(game.state.bag)
    clone.bag.pop()
    assert len(game.state.bag) == original_len


def test_game_state_clone_discard_independent():
    game = Game()
    game.state.discard = [Tile.RED, Tile.BLUE]
    clone = game.state.clone()
    clone.discard.append(Tile.BLACK)
    assert len(game.state.discard) == 2


def test_game_state_clone_players_independent():
    game = Game()
    game.setup_round()
    clone = game.state.clone()
    clone.players[0].score = 99
    assert game.state.players[0].score != 99


# endregion
