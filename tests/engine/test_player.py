# tests/engine/test_player.py
"""Tests for the Player class (engine/player.py)."""

from engine.constants import (
    CAPACITY,
    COLUMN_FOR_TILE_IN_ROW,
    FLOOR,
    FLOOR_PENALTIES,
    WALL_PATTERN,
    Tile,
)
from engine.player import Player

# region Helpers ------------------------------------------------------------


def make_player(**kwargs) -> Player:
    """Return a fresh Player, overriding any fields via kwargs."""
    return Player(name="Test", **kwargs)


def complete_pattern_line(player: Player, row: int, tile: Tile) -> None:
    """Fill a pattern line to capacity with the given tile color."""
    col = COLUMN_FOR_TILE_IN_ROW[tile][row]
    player.pattern_grid[row][col] = CAPACITY[row]


def fill_wall_row(player: Player, row: int) -> None:
    """Place all five tiles on a wall row according to the wall pattern."""
    for col in range(5):
        player.wall[row][col] = 1


def fill_wall_column(player: Player, col: int) -> None:
    """Place all five tiles on a wall column according to the wall pattern."""
    for row in range(5):
        player.wall[row][col] = 1


def fill_wall_tile_color(player: Player, tile: Tile) -> None:
    """Place all five instances of a tile color on the wall."""
    for row in range(5):
        col = COLUMN_FOR_TILE_IN_ROW[tile][row]
        player.wall[row][col] = 1


# endregion


# region earned -------------------------------------------------------------


def test_earned_is_zero_for_fresh_player():
    assert make_player().earned == 0


def test_earned_sums_all_four_components():
    player = make_player(score=10, pending=3, penalty=-2, bonus=5)
    assert player.earned == 16


def test_earned_can_be_negative():
    player = make_player(score=0, pending=0, penalty=-5, bonus=0)
    assert player.earned == -5


# endregion


# region update_score -------------------------------------------------------


def test_update_score_commits_pending_and_penalty():
    player = make_player(score=10, pending=5, penalty=-2, bonus=7)
    player.update_score()
    assert player.score == 13


def test_update_score_resets_pending_and_penalty():
    player = make_player(score=10, pending=5, penalty=-2, bonus=7)
    player.update_score()
    assert player.pending == 0
    assert player.penalty == 0


def test_update_score_clamps_to_zero():
    player = make_player(score=0, pending=0, penalty=-5, bonus=7)
    player.update_score()
    assert player.score == 0


def test_update_score_leaves_bonus_untouched():
    player = make_player(score=5, pending=2, penalty=0, bonus=7)
    player.update_score()
    assert player.bonus == 7


# endregion


# region is_tile_valid_for_row ----------------------------------------------


def test_is_tile_valid_for_row_empty_line_accepts_any_color():
    """An empty pattern line (all zeros) accepts any color."""
    player = make_player()
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is True
    assert player.is_tile_valid_for_row(Tile.RED, 0) is True


def test_is_tile_valid_for_row_rejects_when_wall_cell_already_filled():
    """Wall cell is already filled — cannot place here."""
    player = make_player()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    player.wall[0][col] = 1
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False


def test_is_tile_valid_for_row_rejects_when_line_is_full():
    """Pattern line is full (at capacity) — cannot add more."""
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False


def test_is_tile_valid_for_row_rejects_wrong_color_when_committed():
    """Pattern line is committed to one color, reject wrong color."""
    player = make_player()
    col_blue = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][1]
    player.pattern_grid[1][col_blue] = 1
    # Row 1 is now committed to BLUE, YELLOW is invalid
    assert player.is_tile_valid_for_row(Tile.YELLOW, 1) is False


def test_is_tile_valid_for_row_accepts_matching_color_with_space():
    """Pattern line committed to color with room remaining — accept."""
    player = make_player()
    col_blue = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][1]
    player.pattern_grid[1][col_blue] = 1  # Row 1 capacity is 2
    # BLUE matches and has space (1 of 2 filled)
    assert player.is_tile_valid_for_row(Tile.BLUE, 1) is True


# endregion


# region place --------------------------------------------------------------


def test_place_fills_pattern_line():
    """place() increments the count in the pattern grid."""
    player = make_player()
    player.place(0, [Tile.BLUE])
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == 1


def test_place_overflow_goes_to_floor():
    """When pattern line overflows, excess tiles go to floor."""
    player = make_player()
    # Row 0 has capacity 1 — sending 2 tiles means 1 overflows
    player.place(0, [Tile.BLUE, Tile.BLUE])
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == CAPACITY[0]
    assert Tile.BLUE in player.floor_line


def test_place_to_floor_destination_puts_all_tiles_on_floor():
    """Destination FLOOR places all tiles on floor_line."""
    player = make_player()
    player.place(FLOOR, [Tile.RED, Tile.RED])
    assert player.floor_line == [Tile.RED, Tile.RED]


def test_place_first_player_always_goes_to_floor():
    """FIRST_PLAYER tile is separated and always goes to floor."""
    player = make_player()
    player.place(0, [Tile.FIRST_PLAYER, Tile.BLUE])
    assert Tile.FIRST_PLAYER in player.floor_line
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == 1


def test_place_updates_penalty_cache():
    """Placing tiles on floor updates the penalty cache."""
    player = make_player()
    player.place(FLOOR, [Tile.RED])
    # 1 tile on floor → penalty is FLOOR_PENALTIES[1]
    assert player.penalty == FLOOR_PENALTIES[1]


def test_place_updates_pending_when_line_completes():
    """Completing a pattern line updates pending score."""
    player = make_player()
    # Row 0 has capacity 1 — one tile completes it
    player.place(0, [Tile.BLUE])
    assert player.pending > 0


def test_place_does_not_update_pending_for_incomplete_line():
    """Incomplete pattern line does not trigger pending update."""
    player = make_player()
    # Row 2 has capacity 3 — place only 1
    player.place(2, [Tile.RED])
    assert player.pending == 0


def test_place_first_player_to_floor_destination():
    """Destination FLOOR with FIRST_PLAYER places it correctly."""
    player = make_player()
    player.place(FLOOR, [Tile.FIRST_PLAYER])
    assert player.floor_line == [Tile.FIRST_PLAYER]


def test_place_first_player_alone_no_color_tiles():
    """Placing only FIRST_PLAYER (no color tiles) goes to floor only."""
    player = make_player()
    player.place(0, [Tile.FIRST_PLAYER])
    assert player.floor_line == [Tile.FIRST_PLAYER]
    # Pattern line should remain empty (no color tiles were sent)
    for col in range(5):
        assert player.pattern_grid[0][col] == 0


def test_place_first_player_with_multiple_color_tiles():
    """FIRST_PLAYER + color tiles: color fills line, FIRST_PLAYER to floor."""
    player = make_player()
    # Row 1 has capacity 2
    player.place(1, [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE])
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][1]
    assert player.pattern_grid[1][col] == CAPACITY[1]
    assert player.floor_line == [Tile.FIRST_PLAYER]


def test_place_updates_pending_with_adjacency():
    """Completing a pattern line updates pending with adjacency score."""
    player = make_player()
    # Center tile is WALL_PATTERN[2][2]
    center_tile = WALL_PATTERN[2][2]
    # Pre-fill the four orthogonal neighbors on the wall
    player.wall[1][2] = 1
    player.wall[2][1] = 1
    player.wall[2][3] = 1
    player.wall[3][2] = 1
    # Row 2 has capacity 3 — place 3 to complete
    player.place(2, [center_tile] * 3)
    assert player.pending > 0


# endregion


# region process_round_end --------------------------------------------------


def test_process_round_end_moves_full_line_to_wall():
    """process_round_end() places completed lines on the wall."""
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    player.process_round_end()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert player.wall[0][col] == 1


def test_process_round_end_clears_completed_pattern_line():
    """process_round_end() clears the pattern line after placement."""
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    player.process_round_end()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert player.pattern_grid[0][col] == 0


def test_process_round_end_leaves_incomplete_line_alone():
    """process_round_end() leaves incomplete lines untouched."""
    player = make_player()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.RED][2]
    player.pattern_grid[2][col] = 1
    player.process_round_end()
    assert player.pattern_grid[2][col] == 1


def test_process_round_end_returns_extras_for_discard():
    """process_round_end() returns overflow tiles for discard."""
    player = make_player()
    # Row 1 has capacity 2 — complete it, get 1 overflow
    complete_pattern_line(player, 1, Tile.YELLOW)
    discard = player.process_round_end()
    assert discard.count(Tile.YELLOW) == 1


def test_process_round_end_excludes_first_player_from_discard():
    """process_round_end() removes FIRST_PLAYER from discard pile."""
    player = make_player()
    player.floor_line = [Tile.FIRST_PLAYER, Tile.BLUE]
    discard = player.process_round_end()
    assert Tile.FIRST_PLAYER not in discard
    assert Tile.BLUE in discard


def test_process_round_end_clears_floor():
    """process_round_end() clears the floor_line."""
    player = make_player()
    player.floor_line = [Tile.RED]
    player.process_round_end()
    assert player.floor_line == []


def test_process_round_end_commits_score():
    """process_round_end() adds pending and penalty to score."""
    player = make_player(score=5)
    player.place(0, [Tile.BLUE])
    player.process_round_end()
    assert player.score > 5


# endregion


# region has_triggered_game_end ---------------------------------------------


def test_has_triggered_game_end_false_for_fresh_player():
    assert make_player().has_triggered_game_end() is False


def test_has_triggered_game_end_true_when_wall_row_complete():
    player = make_player()
    fill_wall_row(player, 0)
    assert player.has_triggered_game_end() is True


def test_has_triggered_game_end_true_when_row_pending():
    """Game end triggered when a row is complete (including pending)."""
    player = make_player()
    # Fill 4 of 5 cells in row 0, then complete the missing tile via pattern line
    for col in range(4):
        player.wall[0][col] = 1
    missing_tile = WALL_PATTERN[0][4]
    complete_pattern_line(player, 0, missing_tile)
    assert player.has_triggered_game_end() is True


def test_has_triggered_game_end_false_with_incomplete_row():
    """Game end not triggered if no complete row."""
    player = make_player()
    # Only 4 of 5 cells filled, no pending
    for col in range(4):
        player.wall[0][col] = 1
    assert player.has_triggered_game_end() is False


# endregion


# region encode_completion_progress -----------------------------------------


def test_encode_completion_progress_returns_three_lists():
    result = make_player().encode_completion_progress()
    assert len(result) == 3


def test_encode_completion_progress_each_list_has_five_entries():
    result = make_player().encode_completion_progress()
    assert all(len(group) == 5 for group in result)


def test_encode_completion_progress_all_zero_for_fresh_player():
    result = make_player().encode_completion_progress()
    assert all(value == 0.0 for group in result for value in group)


def test_encode_completion_progress_full_wall_row_is_one():
    player = make_player()
    fill_wall_row(player, 0)
    rows, _cols, _tiles = player.encode_completion_progress()
    assert rows[0] == 1.0


def test_encode_completion_progress_partial_progress():
    """Partial wall row progress is between 0 and 1."""
    player = make_player()
    # Row 2 has capacity 3 — place 1 tile via pattern line
    col = COLUMN_FOR_TILE_IN_ROW[Tile.RED][2]
    player.pattern_grid[2][col] = 1
    rows, _cols, _tiles = player.encode_completion_progress()
    assert 0.0 < rows[2] < 1.0


def test_encode_completion_progress_full_wall_column():
    """Full wall column has completion progress 1.0."""
    player = make_player()
    fill_wall_column(player, 0)
    _rows, cols, _tiles = player.encode_completion_progress()
    assert cols[0] == 1.0


def test_encode_completion_progress_with_adjacency():
    """Adjacency walks in _adjacency_count work without error."""
    player = make_player()
    player.wall[0][0] = 1
    player.wall[1][0] = 1
    # Complete a pattern line for row 2 (which has WALL_PATTERN[2][0] as target)
    tile_at_row2_col0 = WALL_PATTERN[2][0]
    complete_pattern_line(player, 2, tile_at_row2_col0)
    result = player.encode_completion_progress()
    assert result is not None


def test_encode_completion_progress_placed_wall_cell_counts_as_full():
    """A placed wall cell counts as full capacity in _cell_completion."""
    player = make_player()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    player.wall[0][col] = 1
    player.place(1, [Tile.BLUE])
    rows, cols, tiles = player.encode_completion_progress()
    assert all(0.0 <= v <= 1.0 for v in tiles)


# endregion


# region clone --------------------------------------------------------------


def test_clone_preserves_score_components():
    player = make_player(score=7, pending=2, penalty=-1, bonus=3)
    clone = player.clone()
    assert clone.score == player.score
    assert clone.pending == player.pending
    assert clone.penalty == player.penalty
    assert clone.bonus == player.bonus


def test_clone_wall_is_independent():
    player = make_player()
    clone = player.clone()
    clone.wall[0][0] = 1
    assert player.wall[0][0] == 0


def test_clone_pattern_grid_is_independent():
    player = make_player()
    clone = player.clone()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.RED][0]
    clone.pattern_grid[0][col] = 1
    assert player.pattern_grid[0][col] == 0


def test_clone_floor_line_is_independent():
    player = make_player()
    clone = player.clone()
    clone.floor_line.append(Tile.YELLOW)
    assert player.floor_line == []


def test_clone_preserves_name_and_agent():
    player = Player(name="Alice", agent="alphabeta_hard")
    clone = player.clone()
    assert clone.name == "Alice"
    assert clone.agent == "alphabeta_hard"


# endregion


# region __str__ ------------------------------------------------------------


def test_str_contains_player_name():
    player = Player(name="Alice", agent="human")
    assert "Alice" in str(player)


def test_str_contains_agent():
    player = Player(name="Alice", agent="alphabeta_hard")
    assert "alphabeta_hard" in str(player)


def test_str_is_multiline():
    player = make_player()
    assert "\n" in str(player)


def test_repr_is_same_as_str():
    player = Player(name="Alice", agent="human")
    assert repr(player) == str(player)


# endregion
