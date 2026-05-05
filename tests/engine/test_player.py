# tests/test_player.py
"""Tests for the Player class (engine/player.py)."""

import pytest

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
    player.pattern_lines[row] = [tile] * CAPACITY[row]


def fill_wall_row(player: Player, row: int) -> None:
    """Place all five tiles on a wall row according to the wall pattern."""
    for col in range(5):
        player.wall[row][col] = WALL_PATTERN[row][col]


def fill_wall_column(player: Player, col: int) -> None:
    """Place all five tiles on a wall column according to the wall pattern."""
    for row in range(5):
        player.wall[row][col] = WALL_PATTERN[row][col]


def fill_wall_tile_color(player: Player, tile: Tile) -> None:
    """Place all five instances of a tile color on the wall."""
    for row in range(5):
        col = COLUMN_FOR_TILE_IN_ROW[tile][row]
        player.wall[row][col] = tile


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
    player = make_player()
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is True


def test_is_tile_valid_for_row_rejects_when_wall_cell_already_filled():
    player = make_player()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    player.wall[0][col] = Tile.BLUE
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False


def test_is_tile_valid_for_row_rejects_when_line_is_full():
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False


def test_is_tile_valid_for_row_rejects_wrong_color():
    player = make_player()
    player.pattern_lines[1] = [Tile.BLUE]
    assert player.is_tile_valid_for_row(Tile.YELLOW, 1) is False


def test_is_tile_valid_for_row_accepts_matching_color():
    player = make_player()
    player.pattern_lines[1] = [Tile.BLUE]
    assert player.is_tile_valid_for_row(Tile.BLUE, 1) is True


# endregion


# region place --------------------------------------------------------------


def test_place_fills_pattern_line():
    player = make_player()
    player.place(0, [Tile.BLUE])
    assert player.pattern_lines[0] == [Tile.BLUE]


def test_place_overflow_goes_to_floor():
    player = make_player()
    # Row 0 holds only 1 tile — 2 tiles means 1 overflows
    player.place(0, [Tile.BLUE, Tile.BLUE])
    assert player.pattern_lines[0] == [Tile.BLUE]
    assert Tile.BLUE in player.floor_line


def test_place_to_floor_destination_puts_all_tiles_on_floor():
    player = make_player()
    player.place(FLOOR, [Tile.RED, Tile.RED])
    assert player.floor_line == [Tile.RED, Tile.RED]


def test_place_first_player_always_goes_to_floor():
    player = make_player()
    player.place(0, [Tile.FIRST_PLAYER, Tile.BLUE])
    assert Tile.FIRST_PLAYER in player.floor_line
    assert player.pattern_lines[0] == [Tile.BLUE]


def test_place_updates_penalty_cache():
    player = make_player()
    player.place(FLOOR, [Tile.RED])
    assert player.penalty == FLOOR_PENALTIES[0]


def test_place_updates_pending_when_line_completes():
    player = make_player()
    # Row 0 needs exactly 1 tile to complete
    player.place(0, [Tile.BLUE])
    assert player.pending > 0


def test_place_does_not_update_pending_for_incomplete_line():
    player = make_player()
    # Row 2 needs 3 tiles — place only 1
    player.place(2, [Tile.RED])
    assert player.pending == 0


def test_place_first_player_to_floor_destination_lands_on_floor():
    player = make_player()
    player.place(FLOOR, [Tile.FIRST_PLAYER])
    assert player.floor_line == [Tile.FIRST_PLAYER]


def test_place_first_player_alone_with_no_color_tiles():
    player = make_player()
    player.place(0, [Tile.FIRST_PLAYER])
    assert player.floor_line == [Tile.FIRST_PLAYER]
    assert player.pattern_lines[0] == []


def test_place_first_player_with_multiple_color_tiles_all_colors_fill_line():
    player = make_player()
    # Row 1 holds 2 tiles — send 2 color tiles plus FIRST_PLAYER
    player.place(1, [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE])
    assert player.pattern_lines[1] == [Tile.BLUE, Tile.BLUE]
    assert player.floor_line == [Tile.FIRST_PLAYER]


def test_place_updates_pending_with_adjacency():
    player = make_player()
    # Find which tile goes at (2,2) in the wall pattern
    center_tile = WALL_PATTERN[2][2]
    # Pre-fill the four orthogonal neighbors
    player.wall[1][2] = WALL_PATTERN[1][2]
    player.wall[2][1] = WALL_PATTERN[2][1]
    player.wall[2][3] = WALL_PATTERN[2][3]
    player.wall[3][2] = WALL_PATTERN[3][2]
    # Row 2 has capacity 3 — place all 3 to complete the line
    player.place(2, [center_tile] * 3)
    assert player.pending == 6


# endregion


# region process_round_end --------------------------------------------------


def test_process_round_end_moves_full_line_to_wall():
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    player.process_round_end()
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    assert player.wall[0][col] == Tile.BLUE


def test_process_round_end_clears_completed_pattern_line():
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    player.process_round_end()
    assert player.pattern_lines[0] == []


def test_process_round_end_leaves_incomplete_line_alone():
    player = make_player()
    player.pattern_lines[2] = [Tile.RED]
    player.process_round_end()
    assert player.pattern_lines[2] == [Tile.RED]


def test_process_round_end_returns_extras_for_discard():
    player = make_player()
    # Row 1 holds 2 tiles — 1 goes to wall, 1 goes to discard
    complete_pattern_line(player, 1, Tile.YELLOW)
    discard = player.process_round_end()
    assert discard.count(Tile.YELLOW) == 1


def test_process_round_end_excludes_first_player_from_discard():
    player = make_player()
    player.floor_line = [Tile.FIRST_PLAYER, Tile.BLUE]
    discard = player.process_round_end()
    assert Tile.FIRST_PLAYER not in discard
    assert Tile.BLUE in discard


def test_process_round_end_clears_floor():
    player = make_player()
    player.floor_line = [Tile.RED]
    player.process_round_end()
    assert player.floor_line == []


def test_process_round_end_commits_score():
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
    player = make_player()
    # Fill 4 of 5 wall cells in row 0, pending line completes the 5th
    for col in range(4):
        player.wall[0][col] = WALL_PATTERN[0][col]
    missing_tile = WALL_PATTERN[0][4]
    complete_pattern_line(player, 0, missing_tile)
    assert player.has_triggered_game_end() is True


def test_has_triggered_game_end_false_with_incomplete_row():
    player = make_player()
    # Only 4 of 5 cells filled, no pending line
    for col in range(4):
        player.wall[0][col] = WALL_PATTERN[0][col]
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


def test_encode_completion_progress_partial_progress_is_between_zero_and_one():
    player = make_player()
    # Row 2 has capacity 3 — place 1 tile to get partial progress
    player.pattern_lines[2] = [Tile.RED]
    rows, _cols, _tiles = player.encode_completion_progress()
    assert 0.0 < rows[2] < 1.0


# In encode_completion_progress region


def test_encode_completion_progress_counts_placed_wall_cell_as_full():
    player = make_player()
    # Fill an entire wall column — _cell_completion's wall[row][col] branch
    fill_wall_column(player, 0)
    _rows, cols, _tiles = player.encode_completion_progress()
    assert cols[0] == 1.0


def test_encode_completion_progress_with_adjacent_wall_tiles():
    player = make_player()
    # Place tiles in two adjacent rows of the same column so adjacency
    # walks fire in _adjacency_count
    player.wall[0][0] = WALL_PATTERN[0][0]
    player.wall[1][0] = WALL_PATTERN[1][0]
    complete_pattern_line(player, 2, WALL_PATTERN[2][0])
    result = player.encode_completion_progress()
    assert result is not None  # adjacency walk executed without error


def test_encode_completion_progress_placed_wall_cell_counts_as_full_capacity():
    player = make_player()
    # Place Blue on the wall at row 0 (capacity 1, counts as 1)
    col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    player.wall[0][col] = Tile.BLUE
    # Partially fill row 1 pattern line with Blue (1 of 2)
    player.pattern_lines[1] = [Tile.BLUE]
    _rows, _cols, tiles = player.encode_completion_progress()
    blue_index = list(Tile).index(Tile.BLUE) - 0  # BLUE is first COLOR_TILE
    assert tiles[blue_index] == pytest.approx(2 / 15)


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
    clone.wall[0][0] = Tile.BLUE
    assert player.wall[0][0] is None


def test_clone_pattern_lines_are_independent():
    player = make_player()
    clone = player.clone()
    clone.pattern_lines[0].append(Tile.RED)
    assert player.pattern_lines[0] == []


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
