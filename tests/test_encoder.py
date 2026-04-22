# tests/test_encoder.py
"""Encoder tests — changes and additions for the Phase 8d encoding upgrade.

Replace / add these tests in tests/test_encoder.py:

CHANGED (update existing tests with these bodies):
  - test_spatial_shape
  - test_flat_shape
  - test_score_delta_positive_when_ahead
  - test_score_delta_clamped_at_positive_one
  - test_score_delta_clamped_at_negative_one
  - test_score_delta_flips_with_current_player
  - test_scores_swap_with_current_player  (comment updated, logic same)

ADD (new tests, append to file):
  - All tests in the three new sections below

Also add these imports to the top of test_encoder.py:
  from neural.encoder import (
      OFF_ROUND,
      OFF_DISTINCT_PAIRS,
      BLOCKED_WALL_CHANNEL_MY,
      BLOCKED_WALL_CHANNEL_OPP,
  )
"""

import pytest
from engine.game import Game
from engine.constants import (
    Tile,
    BOARD_SIZE,
    COLOR_TILES,
    COLUMN_FOR_TILE_IN_ROW,
)
from neural.encoder import (
    encode_state,
    SPATIAL_SHAPE,
    FLAT_SIZE,
    NUM_COLORS,
    PATTERN_COL,
    OFF_MY_SCORE,
    OFF_SCORE_DELTA,
    OFF_ROUND,
    OFF_DISTINCT_PAIRS,
    BLOCKED_WALL_CHANNEL_MY,
    BLOCKED_WALL_CHANNEL_OPP,
)


def fresh_game() -> Game:
    g = Game()
    g.setup_round()
    return g


def encode(g: Game):
    return encode_state(g)


# ── Changed: shape tests ───────────────────────────────────────────────────


def test_spatial_shape():
    spatial, _ = encode(fresh_game())
    assert spatial.shape == SPATIAL_SHAPE  # (14, 5, 6)


def test_flat_shape():
    _, flat = encode(fresh_game())
    assert flat.shape == (FLAT_SIZE,)  # (49,)


# ── Changed: score delta divisor is now 50 ────────────────────────────────
# These replace the old score_delta tests that used divisor 20.
# Note: scores are now earned_score_unclamped(board), not board.score.
# Setting board.score directly still works because earned_score_unclamped
# includes board.score + pending placement + floor penalty + bonuses.
# With empty walls and pattern lines, pending placement = 0 and floor = 0,
# so earned_score_unclamped == board.score in these simple cases.


def test_score_delta_positive_when_ahead():
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 10
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_SCORE_DELTA].item() == pytest.approx((30 - 10) / 50)


def test_score_delta_clamped_at_positive_one():
    g = fresh_game()
    g.state.players[0].score = 100
    g.state.players[1].score = 0
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_SCORE_DELTA].item() == pytest.approx(1.0)


def test_score_delta_clamped_at_negative_one():
    g = fresh_game()
    g.state.players[0].score = 0
    g.state.players[1].score = 100
    g.state.current_player = 0
    _, flat = encode(g)
    assert flat[OFF_SCORE_DELTA].item() == pytest.approx(-1.0)


def test_score_delta_flips_with_current_player():
    g = fresh_game()
    g.state.players[0].score = 30
    g.state.players[1].score = 10
    g.state.current_player = 0
    _, f0 = encode(g)
    g.state.current_player = 1
    _, f1 = encode(g)
    assert f0[OFF_SCORE_DELTA].item() == pytest.approx(-f1[OFF_SCORE_DELTA].item())


def test_score_delta_not_yet_clamped_reflects_floor_penalty():
    """earned_score_unclamped includes floor penalty; a heavy floor can make
    the delta negative even when board.score is equal."""
    g = fresh_game()
    g.state.players[0].score = 10
    g.state.players[1].score = 10
    # Give current player 7 floor tiles — max penalty is -14
    g.state.players[0].floor_line = [Tile.BLUE] * 7
    g.state.current_player = 0
    _, flat = encode(g)
    # delta should be negative (floor penalty drags current player down)
    assert flat[OFF_SCORE_DELTA].item() < 0.0


# ── New: earned_score_unclamped drives score features ─────────────────────


def test_my_score_reflects_pending_placement_not_just_board_score():
    """earned_score_unclamped includes pattern line placement points.
    A full row-0 pattern line with BLUE will score at least 1 point
    that isn't yet in board.score.
    """
    g = fresh_game()
    g.state.current_player = 0
    # Row 0 capacity is 1; place BLUE which goes in its wall column
    # blue_col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    # Only add pattern line tile if wall cell is empty (it will be)
    g.state.players[0].pattern_lines[0] = [Tile.BLUE]
    # board.score is 0; earned_score_unclamped should be > 0
    _, flat = encode(g)
    assert flat[OFF_MY_SCORE].item() > 0.0


# ── New: blocked_wall channel ─────────────────────────────────────────────


def test_blocked_wall_channel_constants_are_correct_indices():
    """BLOCKED_WALL_CHANNEL_MY must be channel 6, OPP must be channel 13."""
    assert BLOCKED_WALL_CHANNEL_MY == 6
    assert BLOCKED_WALL_CHANNEL_OPP == 13


def test_blocked_wall_all_zero_when_no_pattern_lines_and_empty_wall():
    """With nothing placed anywhere, no cell is blocked."""
    g = Game()
    g.setup_round()
    spatial, _ = encode(g)
    assert spatial[BLOCKED_WALL_CHANNEL_MY].sum().item() == 0.0
    assert spatial[BLOCKED_WALL_CHANNEL_OPP].sum().item() == 0.0


def test_blocked_wall_marks_filled_wall_cells():
    """A filled wall cell is blocked regardless of pattern lines."""
    g = fresh_game()
    g.state.players[0].wall[0][0] = Tile.BLUE
    g.state.current_player = 0
    spatial, _ = encode(g)
    assert spatial[BLOCKED_WALL_CHANNEL_MY, 0, 0].item() == 1.0


def test_blocked_wall_marks_wrong_color_columns_when_pattern_line_committed():
    """If row 1 pattern line is committed to YELLOW, the wall columns for
    all other colors in row 1 should be blocked (if empty)."""
    g = fresh_game()
    g.state.current_player = 0
    # Commit row 1 pattern line to YELLOW
    g.state.players[0].pattern_lines[1] = [Tile.YELLOW]
    yellow_col = COLUMN_FOR_TILE_IN_ROW[Tile.YELLOW][1]
    spatial, _ = encode(g)
    for col in range(BOARD_SIZE):
        cell_value = spatial[BLOCKED_WALL_CHANNEL_MY, 1, col].item()
        if col == yellow_col:
            # The target column is NOT blocked — it's where YELLOW will land
            assert cell_value == 0.0, f"col {col} (YELLOW target) should not be blocked"
        else:
            # Every other column in this row is blocked
            assert cell_value == 1.0, f"col {col} should be blocked"


def test_blocked_wall_does_not_block_already_correct_column():
    """The wall column matching the committed color should never be blocked."""
    g = fresh_game()
    g.state.current_player = 0
    for row in range(BOARD_SIZE):
        color = COLOR_TILES[row % NUM_COLORS]
        g.state.players[0].pattern_lines[row] = [color]
        target_col = COLUMN_FOR_TILE_IN_ROW[color][row]
        spatial, _ = encode(g)
        assert spatial[BLOCKED_WALL_CHANNEL_MY, row, target_col].item() == 0.0


def test_blocked_wall_already_filled_cell_is_blocked():
    """A filled wall cell counts as blocked even if it matches the committed color."""
    g = fresh_game()
    g.state.current_player = 0
    # Fill the BLUE wall cell in row 0
    blue_col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][0]
    g.state.players[0].wall[0][blue_col] = Tile.BLUE
    # Commit row 0 pattern line to BLUE (normally invalid after scoring, but
    # the encoder doesn't enforce game rules — just encode what's there)
    g.state.players[0].pattern_lines[0] = [Tile.BLUE]
    spatial, _ = encode(g)
    # The filled cell is blocked
    assert spatial[BLOCKED_WALL_CHANNEL_MY, 0, blue_col].item() == 1.0


def test_blocked_wall_only_wall_columns_affected_not_pattern_col():
    """The blocked_wall channel should only affect cols 0–4, not col 5."""
    g = fresh_game()
    g.state.current_player = 0
    g.state.players[0].pattern_lines[2] = [Tile.RED, Tile.RED]
    spatial, _ = encode(g)
    # Col 5 (pattern column) must be zero in the blocked channel
    assert spatial[BLOCKED_WALL_CHANNEL_MY, :, PATTERN_COL].sum().item() == 0.0


def test_blocked_wall_opponent_channel_reflects_opponent_board():
    """The opponent blocked_wall channel encodes the opponent's board."""
    g = fresh_game()
    g.state.current_player = 0
    # Only set opponent's pattern line
    g.state.players[1].pattern_lines[0] = [Tile.RED]
    red_col = COLUMN_FOR_TILE_IN_ROW[Tile.RED][0]
    spatial, _ = encode(g)
    # My channel should be zero
    assert spatial[BLOCKED_WALL_CHANNEL_MY].sum().item() == 0.0
    # Opponent channel should have blocked cells in row 0
    for col in range(BOARD_SIZE):
        val = spatial[BLOCKED_WALL_CHANNEL_OPP, 0, col].item()
        if col == red_col:
            assert val == 0.0
        else:
            assert val == 1.0


def test_blocked_wall_swaps_with_current_player():
    """When current_player changes, my/opp blocked channels must swap."""
    g = fresh_game()
    g.state.players[0].pattern_lines[1] = [Tile.BLUE]
    blue_col = COLUMN_FOR_TILE_IN_ROW[Tile.BLUE][1]

    g.state.current_player = 0
    s0, _ = encode(g)
    g.state.current_player = 1
    s1, _ = encode(g)

    # As player 0: my channel row 1 has blocked cells
    non_blue_blocked_as_p0 = sum(
        s0[BLOCKED_WALL_CHANNEL_MY, 1, col].item()
        for col in range(BOARD_SIZE)
        if col != blue_col
    )
    assert non_blue_blocked_as_p0 == pytest.approx(4.0)

    # As player 1: opp channel row 1 has those blocked cells
    non_blue_blocked_as_p1 = sum(
        s1[BLOCKED_WALL_CHANNEL_OPP, 1, col].item()
        for col in range(BOARD_SIZE)
        if col != blue_col
    )
    assert non_blue_blocked_as_p1 == pytest.approx(4.0)

    # And player 1's own channel should be clear (no pattern lines set)
    assert s1[BLOCKED_WALL_CHANNEL_MY].sum().item() == 0.0


# ── New: flat — round progress ────────────────────────────────────────────


def test_round_progress_first_round():
    """Round 1 → (1 - 1) / 5 = 0.0"""
    g = fresh_game()
    g.state.round = 1
    _, flat = encode(g)
    assert flat[OFF_ROUND].item() == pytest.approx(0.0)


def test_round_progress_last_round():
    """Round 6 → (6 - 1) / 5 = 1.0"""
    g = fresh_game()
    g.state.round = 6
    _, flat = encode(g)
    assert flat[OFF_ROUND].item() == pytest.approx(1.0)


def test_round_progress_middle():
    """Round 3 → (3 - 1) / 5 = 0.4"""
    g = fresh_game()
    g.state.round = 3
    _, flat = encode(g)
    assert flat[OFF_ROUND].item() == pytest.approx(0.4)


def test_round_progress_in_range():
    g = fresh_game()
    for round_number in range(1, 7):
        g.state.round = round_number
        _, flat = encode(g)
        assert 0.0 <= flat[OFF_ROUND].item() <= 1.0


# ── New: flat — distinct source-color pairs ───────────────────────────────


def test_distinct_pairs_zero_when_all_sources_empty():
    """With empty factories and center, no pairs remain."""
    g = Game()
    # Don't call setup_round — leave factories and center empty
    _, flat = encode(g)
    assert flat[OFF_DISTINCT_PAIRS].item() == pytest.approx(0.0)


def test_distinct_pairs_counts_factory_colors():
    """A factory with 2 RED and 1 BLUE = 2 distinct pairs for that factory."""
    g = Game()
    g.state.factories[0] = [Tile.RED, Tile.RED, Tile.BLUE]
    _, flat = encode(g)
    # 2 distinct pairs / 10
    assert flat[OFF_DISTINCT_PAIRS].item() == pytest.approx(2 / 10)


def test_distinct_pairs_counts_center_colors():
    """Center with RED and YELLOW = 2 distinct pairs."""
    g = Game()
    g.state.center = [Tile.RED, Tile.YELLOW]
    _, flat = encode(g)
    assert flat[OFF_DISTINCT_PAIRS].item() == pytest.approx(2 / 10)


def test_distinct_pairs_first_player_token_not_counted():
    """FIRST_PLAYER token in center should not contribute a pair."""
    g = Game()
    g.state.center = [Tile.FIRST_PLAYER]
    _, flat = encode(g)
    assert flat[OFF_DISTINCT_PAIRS].item() == pytest.approx(0.0)


def test_distinct_pairs_same_color_across_sources_counted_separately():
    """RED in factory 0 and RED in factory 1 = 2 distinct pairs (different sources)."""
    g = Game()
    g.state.factories[0] = [Tile.RED, Tile.RED]
    g.state.factories[1] = [Tile.RED]
    _, flat = encode(g)
    assert flat[OFF_DISTINCT_PAIRS].item() == pytest.approx(2 / 10)


def test_distinct_pairs_same_color_in_factory_and_center_counted_separately():
    """RED in factory 0 and RED in center = 2 distinct pairs."""
    g = Game()
    g.state.factories[0] = [Tile.RED]
    g.state.center = [Tile.RED]
    _, flat = encode(g)
    assert flat[OFF_DISTINCT_PAIRS].item() == pytest.approx(2 / 10)


def test_distinct_pairs_full_fresh_round_is_reasonable():
    """After setup, there should be substantially more than 0 and <=10 pairs."""
    g = fresh_game()
    _, flat = encode(g)
    raw_count = flat[OFF_DISTINCT_PAIRS].item() * 10
    # A 5-factory game with 4 tiles each, 2 players = 5 factories.
    # Each factory typically has 2-4 colors. Plus center starts with
    # FIRST_PLAYER only. Expect roughly 8-15 distinct pairs.
    assert raw_count > 0.0
    # We don't clamp, so it could exceed 1.0 early in the round — that's fine.
    # Just confirm it's a sensible positive number.
    assert raw_count <= 25.0  # absolute ceiling: 5 factories × 5 colors
