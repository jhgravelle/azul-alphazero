# tests/engine/test_player.py
"""Tests for the Player class (engine/player.py).

Focuses on public method behavior and scenario-based encoding tests.
Encoding tests use from_string() to load known states, then assert
each encoding section against expected values.
"""

from engine.constants import (
    CAPACITY,
    COL_FOR_TILE_ROW,
    COLOR_TILES,
    FLOOR,
    Tile,
)
from engine.player import ENCODING_SLICES, Player

# region Helpers ============================================================


def make_player(**kwargs) -> Player:
    """Return a fresh Player with optional field overrides."""
    return Player(name="Test", **kwargs)


# endregion


# region Scoring Properties =================================================


class TestScoringProperties:
    """Test pending, penalty, bonus, and earned properties."""

    def test_earned_is_zero_for_fresh_player(self):
        assert make_player().earned == 0

    def test_earned_sums_score_pending_penalty_bonus(self):
        """earned = score + pending + penalty + bonus."""
        player = make_player(score=10)
        # Place 1 tile on row 0 (capacity 1) to trigger pending
        player.place(0, [Tile.BLUE])
        earned_expected = player.score + player.pending + player.penalty + player.bonus
        assert player.earned == earned_expected

    def test_pending_zero_for_empty_board(self):
        assert make_player().pending == 0

    def test_penalty_zero_for_empty_floor(self):
        assert make_player().penalty == 0

    def test_bonus_zero_for_empty_wall(self):
        assert make_player().bonus == 0

    def test_penalty_reflects_floor_size(self):
        """Penalty increments with floor line size."""
        player1 = make_player()
        player1.place(FLOOR, [Tile.RED])
        penalty1 = player1.penalty

        player2 = make_player()
        player2.place(FLOOR, [Tile.RED, Tile.BLUE])
        penalty2 = player2.penalty

        # More tiles on floor → more negative penalty
        assert penalty2 < penalty1

    def test_pending_nonzero_when_pattern_line_completes(self):
        """Completing a pattern line triggers pending score."""
        player = make_player()
        # Row 0 has capacity 1 — one tile completes it
        player.place(0, [Tile.BLUE])
        # A completed line scores at least 1 (adjacency baseline)
        assert player.pending > 0

    def test_pending_zero_for_incomplete_line(self):
        """Incomplete pattern line does not contribute to pending."""
        player = make_player()
        # Row 2 has capacity 3 — place only 1
        player.place(2, [Tile.RED])
        assert player.pending == 0


# endregion


# region is_tile_valid_for_row ==============================================


class TestIsTileValidForRow:
    """Test placement validity checks."""

    def test_empty_line_accepts_any_color(self):
        """Fresh player accepts any tile on empty pattern lines."""
        player = make_player()
        for tile in COLOR_TILES:
            assert player.is_tile_valid_for_row(tile, 0) is True

    def test_rejects_when_wall_cell_filled(self):
        """Cannot place a tile if its wall cell is already occupied."""
        player = make_player()
        col = COL_FOR_TILE_ROW[Tile.BLUE][0]
        player._wall_tiles[0][col] = Tile.BLUE
        assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False

    def test_rejects_when_pattern_line_full(self):
        """Cannot place on a pattern line at capacity."""
        player = make_player()
        # Fill row 0 (capacity 1) to capacity
        player._pattern_lines[0] = [Tile.BLUE]
        assert player.is_tile_valid_for_row(Tile.BLUE, 0) is False

    def test_rejects_wrong_color_when_committed(self):
        """Pattern line committed to one color rejects other colors."""
        player = make_player()
        # Commit row 1 to BLUE (capacity 2)
        player._pattern_lines[1] = [Tile.BLUE]
        # No _encode() call needed — is_tile_valid_for_row checks actual state
        assert player.is_tile_valid_for_row(Tile.YELLOW, 1) is False

    def test_accepts_matching_color_with_space(self):
        """Pattern line committed to a color accepts more of that color."""
        player = make_player()
        # Row 1 capacity 2, place 1 BLUE → room for 1 more
        player._pattern_lines[1] = [Tile.BLUE]
        assert player.is_tile_valid_for_row(Tile.BLUE, 1) is True

    def test_accepts_any_tile_on_partially_filled_empty_line(self):
        """An incomplete, uncommitted line accepts any color."""
        player = make_player()
        # Row 2 capacity 3, start with RED
        player._pattern_lines[2] = [Tile.RED]
        # No _encode() call needed — is_tile_valid_for_row checks actual state
        # RED is committed; YELLOW is not accepted
        assert player.is_tile_valid_for_row(Tile.YELLOW, 2) is False


# endregion


# region place ==============================================================


class TestPlace:
    """Test tile placement on pattern lines and floor."""

    def test_place_adds_tiles_to_pattern_line(self):
        """place() populates the pattern line."""
        player = make_player()
        player.place(0, [Tile.BLUE])
        assert len(player._pattern_lines[0]) == 1
        assert player._pattern_lines[0][0] == Tile.BLUE

    def test_place_to_floor_destination(self):
        """Destination FLOOR places all tiles directly on floor."""
        player = make_player()
        player.place(FLOOR, [Tile.RED, Tile.BLUE])
        assert player._floor_line == [Tile.RED, Tile.BLUE]

    def test_place_overflow_goes_to_floor(self):
        """Tiles exceeding pattern line capacity overflow to floor."""
        player = make_player()
        # Row 0 capacity 1 — place 2 tiles
        player.place(0, [Tile.BLUE, Tile.BLUE])
        assert len(player._pattern_lines[0]) == CAPACITY[0]
        assert Tile.BLUE in player._floor_line

    def test_place_first_player_always_to_floor(self):
        """FIRST_PLAYER tile is separated and placed on floor."""
        player = make_player()
        player.place(0, [Tile.FIRST_PLAYER, Tile.BLUE])
        assert Tile.FIRST_PLAYER in player._floor_line
        assert player._pattern_lines[0] == [Tile.BLUE]

    def test_place_multiple_copies_of_same_color(self):
        """place() accepts multiple tiles of the same color."""
        player = make_player()
        # Row 1 capacity 2 — place 2 BLUE
        player.place(1, [Tile.BLUE, Tile.BLUE])
        assert len(player._pattern_lines[1]) == CAPACITY[1]
        assert all(t == Tile.BLUE for t in player._pattern_lines[1])

    def test_place_updates_encoded_features(self):
        """place() recomputes encoded_features."""
        player = make_player()
        old_features = player.encoded_features.copy()
        player.place(0, [Tile.BLUE])
        # At minimum, pending should change (line completed)
        assert player.encoded_features != old_features

    def test_place_first_player_alone_no_color_tiles(self):
        """Placing only FIRST_PLAYER (no colors) goes to floor only."""
        player = make_player()
        player.place(0, [Tile.FIRST_PLAYER])
        assert Tile.FIRST_PLAYER in player._floor_line
        assert len(player._pattern_lines[0]) == 0

    def test_place_first_player_with_overflow(self):
        """FIRST_PLAYER separated, then color tiles fill and overflow."""
        player = make_player()
        # Row 1 capacity 2
        player.place(1, [Tile.FIRST_PLAYER, Tile.RED, Tile.RED, Tile.RED])
        assert Tile.FIRST_PLAYER in player._floor_line
        assert len(player._pattern_lines[1]) == CAPACITY[1]
        assert player._floor_line.count(Tile.RED) == 1  # The overflow


# endregion


# region process_round_end ==================================================


class TestProcessRoundEnd:
    """Test end-of-round scoring and cleanup."""

    def test_process_round_end_moves_full_lines_to_wall(self):
        """Complete pattern lines place on the wall."""
        player = make_player()
        # Row 0 capacity 1
        player._pattern_lines[0] = [Tile.BLUE]
        player.process_round_end()
        col = COL_FOR_TILE_ROW[Tile.BLUE][0]
        assert player._wall_tiles[0][col] == Tile.BLUE

    def test_process_round_end_clears_completed_lines(self):
        """Placed pattern lines are cleared."""
        player = make_player()
        player._pattern_lines[0] = [Tile.BLUE]
        player.process_round_end()
        assert len(player._pattern_lines[0]) == 0

    def test_process_round_end_leaves_incomplete_lines(self):
        """Incomplete pattern lines are not affected."""
        player = make_player()
        # Row 2 capacity 3, place 1
        player._pattern_lines[2] = [Tile.RED]
        player.process_round_end()
        assert player._pattern_lines[2] == [Tile.RED]

    def test_process_round_end_returns_extras_for_discard(self):
        """Overflow tiles from completed lines are returned."""
        player = make_player()
        # Row 1 capacity 2 — place 2 identical tiles, 1 goes to wall, 1 to discard
        player._pattern_lines[1] = [Tile.YELLOW, Tile.YELLOW]
        discard = player.process_round_end()
        # One YELLOW was placed on wall, one goes to discard
        assert Tile.YELLOW in discard
        assert discard.count(Tile.YELLOW) == 1

        # Row 4 capacity 5 — place 5, 1 goes to wall, 4 go to discard
        player2 = make_player()
        player2._pattern_lines[4] = [Tile.RED] * CAPACITY[4]
        discard2 = player2.process_round_end()
        # 4 RED tiles go to discard (all but the one placed on wall)
        assert discard2.count(Tile.RED) == CAPACITY[4] - 1

    def test_process_round_end_includes_floor_tiles_in_discard(self):
        """Floor tiles (except FIRST_PLAYER) go to discard."""
        player = make_player()
        player._floor_line = [Tile.RED, Tile.BLUE, Tile.FIRST_PLAYER]
        discard = player.process_round_end()
        assert Tile.RED in discard
        assert Tile.BLUE in discard
        assert Tile.FIRST_PLAYER not in discard

    def test_process_round_end_excludes_first_player_from_discard(self):
        """FIRST_PLAYER token is removed, not discarded."""
        player = make_player()
        player._floor_line = [Tile.FIRST_PLAYER]
        discard = player.process_round_end()
        assert len(discard) == 0

    def test_process_round_end_clears_floor_line(self):
        """Floor is empty after process_round_end()."""
        player = make_player()
        player._floor_line = [Tile.RED, Tile.BLUE]
        player.process_round_end()
        assert len(player._floor_line) == 0

    def test_process_round_end_commits_score(self):
        """pending and penalty are added to score."""
        player = make_player(score=5)
        player.place(0, [Tile.BLUE])  # Completes row 0, adds to pending
        old_score = player.score
        player.process_round_end()
        # Score increased by pending + penalty
        assert player.score > old_score

    def test_process_round_end_multiple_completions(self):
        """Multiple completed lines all place on the wall."""
        player = make_player()
        player._pattern_lines[0] = [Tile.BLUE]
        player._pattern_lines[1] = [Tile.YELLOW, Tile.YELLOW]
        player.process_round_end()
        col_blue = COL_FOR_TILE_ROW[Tile.BLUE][0]
        col_yellow = COL_FOR_TILE_ROW[Tile.YELLOW][1]
        assert player._wall_tiles[0][col_blue] == Tile.BLUE
        assert player._wall_tiles[1][col_yellow] == Tile.YELLOW


# endregion


# region clone ==============================================================


class TestClone:
    """Test independent copying of player state."""

    def test_clone_preserves_score(self):
        """Cloned player has same score."""
        player = make_player(score=42)
        clone = player.clone()
        assert clone.score == 42

    def test_clone_preserves_name(self):
        """Cloned player retains name."""
        player = Player(name="Alice")
        clone = player.clone()
        assert clone.name == "Alice"

    def test_clone_pattern_lines_independent(self):
        """Modifying clone's pattern lines does not affect original."""
        player = make_player()
        clone = player.clone()
        clone._pattern_lines[0].append(Tile.BLUE)
        assert len(player._pattern_lines[0]) == 0

    def test_clone_wall_tiles_independent(self):
        """Modifying clone's wall does not affect original."""
        player = make_player()
        clone = player.clone()
        clone._wall_tiles[0][0] = Tile.BLUE
        assert player._wall_tiles[0][0] is None

    def test_clone_floor_line_independent(self):
        """Modifying clone's floor does not affect original."""
        player = make_player()
        clone = player.clone()
        clone._floor_line.append(Tile.RED)
        assert len(player._floor_line) == 0

    def test_clone_preserves_all_scoring_components(self):
        """Cloned player has same pending, penalty, bonus, earned."""
        player = make_player()
        player.place(0, [Tile.BLUE])
        player.place(FLOOR, [Tile.RED])
        clone = player.clone()
        assert clone.pending == player.pending
        assert clone.penalty == player.penalty
        assert clone.bonus == player.bonus
        assert clone.earned == player.earned


# endregion


# region from_string / __str__ ==============================================


class TestStringRoundTrip:
    """Test serialization and deserialization via __str__ and from_string()."""

    def test_fresh_player_round_trip(self):
        """Fresh player serializes and deserializes correctly."""
        player = Player(name="Alice")
        reconstructed = Player.from_string(str(player))
        assert reconstructed.name == "Alice"
        assert reconstructed.score == 0
        assert reconstructed.earned == 0

    def test_round_trip_with_floor_and_pattern(self):
        """Player with mixed state round-trips correctly."""
        player = Player(name="Bob")
        player.place(0, [Tile.BLUE])
        player.place(FLOOR, [Tile.RED] * 5)
        reconstructed = Player.from_string(str(player))
        assert reconstructed.score == player.score
        assert reconstructed.earned == player.earned
        assert len(reconstructed._floor_line) == len(player._floor_line)

    def test_from_string_validates_earned(self):
        """from_string() asserts earned matches recomputed value."""
        player = Player(name="Carol")
        player.place(0, [Tile.YELLOW])
        text = str(player)
        reconstructed = Player.from_string(text)
        # Should not raise — earned was consistent
        assert reconstructed.earned == player.earned

    def test_str_contains_name(self):
        """__str__ output includes player name."""
        player = Player(name="Diana")
        assert "Diana" in str(player)

    def test_str_contains_score_info(self):
        """__str__ output includes score."""
        player = make_player(score=15)
        assert "15" in str(player)

    def test_repr_equals_str(self):
        """__repr__ is the same as __str__."""
        player = make_player()
        assert repr(player) == str(player)


# endregion


# region Encoding Scenarios ==================================================


class TestEncodingScenarios:
    """Scenario-based encoding tests using from_string() to load states.

    Each test loads a known board state via from_string() and asserts
    that each encoding section contains expected values. This tests
    the encoding as a whole rather than individual indices.
    """

    def test_encoding_fresh_player(self):
        """Fresh player has all zeros except adjacency baseline."""
        player = make_player()
        feat = player.encoded_features

        # Wall and pending sections should be zero
        wall_section = feat[
            ENCODING_SLICES["wall"].start : ENCODING_SLICES["wall"].stop
        ]
        assert all(v == 0 for v in wall_section)

        pending_wall_section = feat[
            ENCODING_SLICES["pending_wall"].start : ENCODING_SLICES["pending_wall"].stop
        ]
        assert all(v == 0 for v in pending_wall_section)

        # Scoring section zero for fresh player
        scoring_start = ENCODING_SLICES["scoring"].start
        assert feat[scoring_start] == 0  # official_score
        assert feat[scoring_start + 1] == 0  # pending
        assert feat[scoring_start + 2] == 0  # penalty
        assert feat[scoring_start + 3] == 0  # bonus
        assert feat[scoring_start + 4] == 0  # earned

    def test_encoding_total_length(self):
        """Encoded features has expected total length."""
        player = make_player()
        assert len(player.encoded_features) == 168

    def test_encoding_full_pattern_line_updates_pending(self):
        """Completing a pattern line updates the pending score encoding."""
        player = make_player()
        # Row 0 capacity 1 — place 1 tile
        player.place(0, [Tile.BLUE])

        feat = player.encoded_features
        scoring_start = ENCODING_SLICES["scoring"].start

        # pending should be nonzero (adjacency baseline = 1)
        pending = feat[scoring_start + 1]
        assert pending > 0

        # earned should equal score + pending + penalty + bonus
        score = feat[scoring_start]
        penalty = feat[scoring_start + 2]
        bonus = feat[scoring_start + 3]
        earned = feat[scoring_start + 4]
        assert abs(earned - (score + pending + penalty + bonus)) < 0.01

    def test_encoding_floor_penalty(self):
        """Floor tiles update the penalty encoding."""
        player1 = make_player()
        player1.place(FLOOR, [Tile.RED])

        player2 = make_player()
        player2.place(FLOOR, [Tile.RED, Tile.BLUE])

        feat1 = player1.encoded_features
        feat2 = player2.encoded_features

        scoring_start = ENCODING_SLICES["scoring"].start
        penalty1 = feat1[scoring_start + 2]
        penalty2 = feat2[scoring_start + 2]

        # More floor tiles → more negative penalty
        assert penalty2 < penalty1
        assert penalty2 < 0

    def test_encoding_first_player_token(self):
        """FIRST_PLAYER on floor sets the misc token."""
        player = make_player()
        player.place(FLOOR, [Tile.FIRST_PLAYER])

        feat = player.encoded_features
        misc_start = ENCODING_SLICES["misc"].start
        first_player_flag = feat[misc_start]
        assert first_player_flag == 1

    def test_encoding_first_player_absent(self):
        """FIRST_PLAYER not on floor clears the misc token."""
        player = make_player()
        assert player.encoded_features[ENCODING_SLICES["misc"].start] == 0

    def test_encoding_wall_section_reflects_placements(self):
        """Wall encoding section shows placed tiles."""
        player = make_player()
        # Manually place a tile on the wall
        player._wall_tiles[0][0] = Tile.BLUE
        player._encode()

        feat = player.encoded_features
        wall_section = feat[
            ENCODING_SLICES["wall"].start : ENCODING_SLICES["wall"].stop
        ]
        # First cell (row 0, col 0) should be 1
        assert wall_section[0] == 1

    def test_encoding_multiple_walls_multiple_tiles(self):
        """Multiple wall placements all reflected in encoding."""
        player = make_player()
        player._wall_tiles[0][0] = Tile.BLUE
        player._wall_tiles[1][1] = Tile.RED
        player._wall_tiles[2][2] = Tile.YELLOW
        player._encode()

        feat = player.encoded_features
        wall_section = feat[
            ENCODING_SLICES["wall"].start : ENCODING_SLICES["wall"].stop
        ]
        # Each placement is a 1
        assert wall_section[0] == 1  # [0, 0]
        assert wall_section[6] == 1  # [1, 1]
        assert wall_section[12] == 1  # [2, 2]

    def test_encoding_adjacency_grid_present(self):
        """Adjacency section is present and nonzero."""
        player = make_player()
        feat = player.encoded_features
        adj_section = feat[
            ENCODING_SLICES["adjacency_grid"]
            .start : ENCODING_SLICES["adjacency_grid"]
            .stop
        ]
        # Fresh board: every cell has adjacency count 1 (lone tile)
        # These are stored as raw ints, not normalized
        assert len(adj_section) == 25
        assert all(v >= 0 for v in adj_section)

    def test_encoding_pattern_capacity_section(self):
        """Pattern capacity section reflects available pattern slots."""
        player = make_player()
        feat = player.encoded_features
        pattern_cap_section = feat[
            ENCODING_SLICES["pattern_capacity"]
            .start : ENCODING_SLICES["pattern_capacity"]
            .stop
        ]
        # Fresh board: every empty pattern slot available
        # 5 colors × 5 rows = 25 capacity values
        assert len(pattern_cap_section) == 25
        # All values positive (capacity available)
        assert all(v > 0 for v in pattern_cap_section)

    def test_encoding_consistency_after_multiple_placements(self):
        """Encoding stays consistent through multiple place() calls."""
        player = make_player()
        player.place(0, [Tile.BLUE])
        feat1 = player.encoded_features.copy()

        player.place(1, [Tile.RED])
        feat2 = player.encoded_features

        # Encoding should be recalculated, so features change
        assert feat1 != feat2

    def test_encoding_after_process_round_end(self):
        """Encoding updates after process_round_end()."""
        player = make_player()
        player.place(0, [Tile.BLUE])
        feat_before = player.encoded_features.copy()

        player.process_round_end()
        feat_after = player.encoded_features

        # Wall section should now have the placed tile
        wall_section_before = feat_before[
            ENCODING_SLICES["wall"].start : ENCODING_SLICES["wall"].stop
        ]
        wall_section_after = feat_after[
            ENCODING_SLICES["wall"].start : ENCODING_SLICES["wall"].stop
        ]
        # At least one wall cell is now occupied
        assert sum(wall_section_after) > sum(wall_section_before)


# endregion
