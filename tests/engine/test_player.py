# tests/engine/test_player.py
"""Tests for the Player class (engine/player.py)."""

from engine.constants import (
    CAPACITY,
    COL_FOR_TILE_ROW,
    COLOR_TILES,
    FLOOR,
    FLOOR_PENALTIES,
    FLOOR_SIZE,
    SIZE,
    TILE_FOR_ROW_COL,
    Tile,
)
from engine.player import Player

# region Helpers ------------------------------------------------------------


def make_player(**kwargs) -> Player:
    """Return a fresh Player, overriding any fields via kwargs."""
    return Player(name="Test", **kwargs)


def complete_pattern_line(player: Player, row: int, tile: Tile) -> None:
    """Fill a pattern line to capacity with the given tile color."""
    col = COL_FOR_TILE_ROW[tile][row]
    player._pattern_grid[row][col] = CAPACITY[row]


def fill_wall_row(player: Player, row: int) -> None:
    """Place all five tiles on a wall row according to the wall pattern."""
    for col in range(SIZE):
        player._wall[row][col] = 1


def fill_wall_column(player: Player, col: int) -> None:
    """Place all five tiles on a wall column according to the wall pattern."""
    for row in range(SIZE):
        player._wall[row][col] = 1


def fill_wall_tile_color(player: Player, tile: Tile) -> None:
    """Place all five instances of a tile color on the wall."""
    for row in range(SIZE):
        col = COL_FOR_TILE_ROW[tile][row]
        player._wall[row][col] = 1


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


# region _update_score ------------------------------------------------------


def test_update_score_commits_pending_and_penalty():
    player = make_player(score=10, pending=5, penalty=-2, bonus=7)
    player._update_score()
    assert player.score == 13


def test_update_score_resets_pending_and_penalty():
    player = make_player(score=10, pending=5, penalty=-2, bonus=7)
    player._update_score()
    assert player.pending == 0
    assert player.penalty == 0


def test_update_score_clamps_to_zero():
    player = make_player(score=0, pending=0, penalty=-5, bonus=7)
    player._update_score()
    assert player.score == 0


def test_update_score_leaves_bonus_untouched():
    player = make_player(score=5, pending=2, penalty=0, bonus=7)
    player._update_score()
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
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    player._wall[0][col] = 1
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False


def test_is_tile_valid_for_row_rejects_when_line_is_full():
    """Pattern line is full (at capacity) — cannot add more."""
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False


def test_is_tile_valid_for_row_rejects_wrong_color_when_committed():
    """Pattern line is committed to one color, reject wrong color."""
    player = make_player()
    col_blue = COL_FOR_TILE_ROW[Tile.BLUE][1]
    player._pattern_grid[1][col_blue] = 1
    # Row 1 is now committed to BLUE, YELLOW is invalid
    assert player.is_tile_valid_for_row(Tile.YELLOW, 1) is False


def test_is_tile_valid_for_row_accepts_matching_color_with_space():
    """Pattern line committed to color with room remaining — accept."""
    player = make_player()
    col_blue = COL_FOR_TILE_ROW[Tile.BLUE][1]
    player._pattern_grid[1][col_blue] = 1  # Row 1 capacity is 2
    # BLUE matches and has space (1 of 2 filled)
    assert player.is_tile_valid_for_row(Tile.BLUE, 1) is True


# endregion


# region place --------------------------------------------------------------


def test_place_fills_pattern_line():
    """place() increments the count in the pattern grid."""
    player = make_player()
    player.place(0, [Tile.BLUE])
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player._pattern_grid[0][col] == 1


def test_place_overflow_goes_to_floor():
    """When pattern line overflows, excess tiles go to floor."""
    player = make_player()
    # Row 0 has capacity 1 — sending 2 tiles means 1 overflows
    player.place(0, [Tile.BLUE, Tile.BLUE])
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player._pattern_grid[0][col] == CAPACITY[0]
    assert Tile.BLUE in player._floor_line


def test_place_to_floor_destination_puts_all_tiles_on_floor():
    """Destination FLOOR places all tiles on _floor_line."""
    player = make_player()
    player.place(FLOOR, [Tile.RED, Tile.RED])
    assert player._floor_line == [Tile.RED, Tile.RED]


def test_place_first_player_always_goes_to_floor():
    """FIRST_PLAYER tile is separated and always goes to floor."""
    player = make_player()
    player.place(0, [Tile.FIRST_PLAYER, Tile.BLUE])
    assert Tile.FIRST_PLAYER in player._floor_line
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player._pattern_grid[0][col] == 1


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
    assert player._floor_line == [Tile.FIRST_PLAYER]


def test_place_first_player_alone_no_color_tiles():
    """Placing only FIRST_PLAYER (no color tiles) goes to floor only."""
    player = make_player()
    player.place(0, [Tile.FIRST_PLAYER])
    assert player._floor_line == [Tile.FIRST_PLAYER]
    # Pattern line should remain empty (no color tiles were sent)
    for col in range(SIZE):
        assert player._pattern_grid[0][col] == 0


def test_place_first_player_with_multiple_color_tiles():
    """FIRST_PLAYER + color tiles: color fills line, FIRST_PLAYER to floor."""
    player = make_player()
    # Row 1 has capacity 2
    player.place(1, [Tile.FIRST_PLAYER, Tile.BLUE, Tile.BLUE])
    col = COL_FOR_TILE_ROW[Tile.BLUE][1]
    assert player._pattern_grid[1][col] == CAPACITY[1]
    assert player._floor_line == [Tile.FIRST_PLAYER]


def test_place_updates_pending_with_adjacency():
    """Completing a pattern line updates pending with adjacency score."""
    player = make_player()
    # Center tile is WALL_PATTERN[2][2]
    center_tile = TILE_FOR_ROW_COL[2][2]
    # Pre-fill the four orthogonal neighbors on the wall
    player._wall[1][2] = 1
    player._wall[2][1] = 1
    player._wall[2][3] = 1
    player._wall[3][2] = 1
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
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player._wall[0][col] == 1


def test_process_round_end_clears_completed_pattern_line():
    """process_round_end() clears the pattern line after placement."""
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    player.process_round_end()
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player._pattern_grid[0][col] == 0


def test_process_round_end_leaves_incomplete_line_alone():
    """process_round_end() leaves incomplete lines untouched."""
    player = make_player()
    col = COL_FOR_TILE_ROW[Tile.RED][2]
    player._pattern_grid[2][col] = 1
    player.process_round_end()
    assert player._pattern_grid[2][col] == 1


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
    player._floor_line = [Tile.FIRST_PLAYER, Tile.BLUE]
    discard = player.process_round_end()
    assert Tile.FIRST_PLAYER not in discard
    assert Tile.BLUE in discard


def test_process_round_end_clears_floor():
    """process_round_end() clears the _floor_line."""
    player = make_player()
    player._floor_line = [Tile.RED]
    player.process_round_end()
    assert player._floor_line == []


def test_process_round_end_commits_score():
    """process_round_end() adds pending and penalty to score."""
    player = make_player(score=5)
    player.place(0, [Tile.BLUE])
    player.process_round_end()
    assert player.score > 5


def test_process_round_end_writes_int_to_wall():
    """process_round_end() writes 1 (not True) to the wall — wall is list[list[int]]."""
    player = make_player()
    complete_pattern_line(player, 0, Tile.BLUE)
    player.process_round_end()
    col = COL_FOR_TILE_ROW[Tile.BLUE][0]
    assert player._wall[0][col] == 1
    assert type(player._wall[0][col]) is int


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
        player._wall[0][col] = 1
    missing_tile = TILE_FOR_ROW_COL[0][4]
    complete_pattern_line(player, 0, missing_tile)
    assert player.has_triggered_game_end() is True


def test_has_triggered_game_end_false_with_incomplete_row():
    """Game end not triggered if no complete row."""
    player = make_player()
    # Only 4 of 5 cells filled, no pending
    for col in range(4):
        player._wall[0][col] = 1
    assert player.has_triggered_game_end() is False


# endregion


# region encode -------------------------------------------------------------


def test_encode_returns_150_values():
    assert len(make_player().encode()) == 150


def test_encode_returns_floats():
    encoding = make_player().encode()
    assert all(isinstance(v, float) for v in encoding)


def test_encode_zero_sections_for_fresh_player():
    """Most sections are zero for a fresh player. Adjacency-related sections
    are nonzero — every empty cell has a lone-tile adjacency of 1."""
    encoding = make_player().encode()
    # Wall (25), pattern fills (25), pattern flags (5), scoring (5),
    # first-player (1), wall completion progress (15), top completions (6),
    # tiles needed (1), incomplete count (1), pattern line demand (5)
    assert all(v == 0.0 for v in encoding[0:82])
    # adjacency grid (82..107): nonzero (lone-tile baseline)
    # tiles needed (107) and incomplete count (108) and pattern demand (109..113): zero
    assert all(v == 0.0 for v in encoding[107:114])
    # wall completion demand (114..143) and adjacency demand (144..148): nonzero
    # baseline total used tiles (149): zero
    assert encoding[149] == 0.0


def test_encode_wall_section_reflects_placements():
    """First 25 values are binary wall, row-major."""
    player = make_player()
    fill_wall_row(player, 0)
    encoding = player.encode()
    assert encoding[0:5] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert encoding[5:10] == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_encode_pattern_fill_ratio():
    """Pattern fill ratios in indices 25..49 are fill/CAPACITY."""
    player = make_player()
    # Row 2 capacity 3, place 1 BLUE → ratio 1/3
    col = COL_FOR_TILE_ROW[Tile.BLUE][2]
    player._pattern_grid[2][col] = 1
    encoding = player.encode()
    # row 2, col `col` is at index 25 + 2*5 + col
    idx = 25 + 2 * 5 + col
    assert abs(encoding[idx] - 1 / 3) < 1e-9


def test_encode_pattern_completion_flags():
    """Indices 50..54 are 1.0 if pattern line is full else 0.0."""
    player = make_player()
    complete_pattern_line(player, 2, Tile.BLUE)
    encoding = player.encode()
    flags = encoding[50:55]
    assert flags[2] == 1.0
    assert all(flags[i] == 0.0 for i in (0, 1, 3, 4))


def test_encode_scoring_normalization():
    """Indices 55..59 are score, pending, penalty, bonus, earned each /100."""
    player = make_player(score=50, pending=10, penalty=-3, bonus=7)
    encoding = player.encode()
    assert encoding[55] == 0.50
    assert encoding[56] == 0.10
    assert encoding[57] == -0.03
    assert encoding[58] == 0.07
    assert abs(encoding[59] - (50 + 10 - 3 + 7) / 100) < 1e-9


def test_encode_first_player_token():
    """Index 60 is 1.0 if FIRST_PLAYER on floor, else 0.0."""
    player = make_player()
    assert make_player().encode()[60] == 0.0
    player._floor_line = [Tile.FIRST_PLAYER]
    assert player.encode()[60] == 1.0


def test_encode_wall_completion_progress_full_row_is_one():
    """Indices 61..75 are completion fractions for rows, cols, colors."""
    player = make_player()
    fill_wall_row(player, 0)
    encoding = player.encode()
    # Row 0 is at index 61
    assert encoding[61] == 1.0
    # Other rows still 0
    assert encoding[62] == 0.0


def test_encode_top_completions_descending_per_feature():
    """Indices 76..81 are top 3 row + top 2 col + top 1 color completions, each sorted
    desc."""
    player = make_player()
    fill_wall_row(player, 2)  # row 2 = 1.0 completion
    encoding = player.encode()
    top_completions = encoding[76:82]
    # The top row is 1.0 (the completed one); the next two are 0.0
    assert top_completions[0] == 1.0
    # Top col includes col contributions from row 2 placements
    # Top tile color includes color contributions from row 2 placements
    # Top values for cols/colors are non-trivial; ensure list is descending within each
    # feature
    assert top_completions[0] >= top_completions[1] >= top_completions[2]
    assert top_completions[3] >= top_completions[4]


def test_encode_adjacency_grid_normalization():
    """Indices 82..106 are adjacency_count(row,col)/10 for each wall cell."""
    encoding = make_player().encode()
    grid_section = encoding[82:107]
    # Empty wall: every cell has _adjacency_count == 1 (lone tile)
    assert all(abs(v - 0.1) < 1e-9 for v in grid_section)


def test_encode_tiles_needed_zero_for_fresh():
    assert make_player().encode()[107] == 0.0


def test_encode_tiles_needed_counts_started_incomplete():
    """Tiles needed: started incomplete lines contribute CAPACITY-fill."""
    player = make_player()
    # Row 4 (capacity 5) with 1 BLUE → needs 4
    col = COL_FOR_TILE_ROW[Tile.BLUE][4]
    player._pattern_grid[4][col] = 1
    encoding = player.encode()
    assert abs(encoding[107] - 4 / 10) < 1e-9


def test_encode_tiles_needed_excludes_completed_lines():
    """A full pattern line contributes 0 to tiles needed."""
    player = make_player()
    complete_pattern_line(player, 4, Tile.BLUE)
    encoding = player.encode()
    assert encoding[107] == 0.0


def test_encode_incomplete_lines_count_raw_int():
    """Index 108 is the raw count (as float) of started incomplete lines."""
    player = make_player()
    col0 = COL_FOR_TILE_ROW[Tile.BLUE][0]
    col1 = COL_FOR_TILE_ROW[Tile.BLUE][1]
    player._pattern_grid[0][col0] = 1  # row 0 cap 1 → full, NOT incomplete
    player._pattern_grid[1][col1] = 1  # row 1 cap 2 → started incomplete
    encoding = player.encode()
    assert encoding[108] == 1.0


def test_encode_pattern_line_demand_per_color():
    """Indices 109..113: per color, tiles needed on lines committed to that
    color / 10."""
    player = make_player()
    # Row 4 (cap 5) committed to RED with 2 tiles → demand[RED] = 3
    col = COL_FOR_TILE_ROW[Tile.RED][4]
    player._pattern_grid[4][col] = 2
    encoding = player.encode()
    red_idx = 109 + COLOR_TILES.index(Tile.RED)
    assert abs(encoding[red_idx] - 0.3) < 1e-9
    # Other colors still 0
    for tile in COLOR_TILES:
        if tile == Tile.RED:
            continue
        idx = 109 + COLOR_TILES.index(tile)
        assert encoding[idx] == 0.0


def test_encode_wall_completion_demand_30_values():
    """Indices 114..143 are 6 groups × 5 colors empty cells / 10."""
    encoding = make_player().encode()
    section = encoding[114:144]
    assert len(section) == 30
    # Empty wall: every cell contributes 1 to its color's count
    # Top row group: 5 cells, one per color → each color gets 1, /10 = 0.1
    first_row_group = section[0:5]
    assert all(abs(v - 0.1) < 1e-9 for v in first_row_group)


def test_encode_adjacency_demand_per_color():
    """Indices 144..148 are per-color sum of adjacency over empty cells / 10."""
    encoding = make_player().encode()
    section = encoding[144:149]
    # Fresh player: every cell has adjacency 1, 5 cells per color → sum 5, /10 = 0.5
    assert all(abs(v - 0.5) < 1e-9 for v in section)


def test_encode_total_used_tiles():
    """Index 149: (wall_tiles_weighted + pattern_tiles) / 100."""
    assert make_player().encode()[149] == 0.0
    player = make_player()
    fill_wall_row(player, 4)  # 5 cells in row 4, each weighted CAPACITY[4]=5 → 25 tiles
    encoding = player.encode()
    assert abs(encoding[149] - 25 / 100) < 1e-9


def test_encode_pending_pattern_line_excludes_from_demand():
    """A full pattern line counts as filled — its cell is not 'empty' for demand."""
    player = make_player()
    # Complete row 0 BLUE — that wall cell is now pending
    complete_pattern_line(player, 0, Tile.BLUE)
    encoding = player.encode()
    # Adjacency demand for BLUE excludes that pending cell (pending != empty)
    blue_demand = encoding[144 + COLOR_TILES.index(Tile.BLUE)]
    # Without the pending cell, BLUE has 4 empty cells contributing
    # (each lone with adjacency 1 since wall is otherwise empty)
    # so demand = 4/10 = 0.4
    assert abs(blue_demand - 0.4) < 1e-9


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
    clone._wall[0][0] = 1
    assert player._wall[0][0] == 0


def test_clone_pattern_grid_is_independent():
    player = make_player()
    clone = player.clone()
    col = COL_FOR_TILE_ROW[Tile.RED][0]
    clone._pattern_grid[0][col] = 1
    assert player._pattern_grid[0][col] == 0


def test_clone_floor_line_is_independent():
    player = make_player()
    clone = player.clone()
    clone._floor_line.append(Tile.YELLOW)
    assert player._floor_line == []


def test_clone_preserves_name():
    player = Player(name="Alice")
    clone = player.clone()
    assert clone.name == "Alice"


# endregion


# region __str__ ------------------------------------------------------------


def test_str_contains_player_name():
    player = Player(name="Alice")
    assert "Alice" in str(player)


def test_str_is_multiline():
    player = make_player()
    assert "\n" in str(player)


def test_str_uses_floor_size_constant():
    """Floor display has FLOOR_SIZE penalty slots (currently 7)."""
    player = make_player()
    # All slots empty → 7 dots interspersed with spaces in the slot region
    floor_line = str(player).splitlines()[-1]
    assert floor_line.count(".") == FLOOR_SIZE


def test_repr_is_same_as_str():
    player = Player(name="Alice")
    assert repr(player) == str(player)


# endregion


# region from_string --------------------------------------------------------


def test_from_string_round_trip_fresh_player():
    player = Player(name="Alice")
    reconstructed = Player.from_string(str(player))
    assert reconstructed.name == "Alice"
    assert reconstructed.score == 0
    assert reconstructed.earned == 0


def test_from_string_round_trip_with_state():
    player = Player(name="Bob")
    player.place(3, [Tile.BLACK] * 2)
    player.place(FLOOR, [Tile.BLACK] * 9)
    reconstructed = Player.from_string(str(player))
    assert reconstructed.score == player.score
    assert reconstructed.earned == player.earned
    assert reconstructed._floor_line == player._floor_line


# endregion
