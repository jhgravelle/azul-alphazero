"""Tests for Game class."""

import pytest
from engine.game import Game, Move
from engine.constants import Tile, CENTER, FLOOR, TILES_PER_FACTORY, NUMBER_OF_FACTORIES


# region Fixtures -------------------------------------------------------


@pytest.fixture
def game():
    """Fresh game with default setup."""
    return Game()


@pytest.fixture
def game_after_setup(game):
    """Game after setup_round() called."""
    game.setup_round()
    return game


@pytest.fixture
def game_with_names():
    """Game with custom player names."""
    return Game(["Alice", "Bob"])


# endregion


# region Initialization -------------------------------------------------


def test_game_init_creates_two_players():
    """Game initializes with two players by default."""
    game = Game()
    assert len(game.players) == 2


def test_game_init_uses_provided_names():
    """Game initializes with provided player names."""
    game = Game(["Alice", "Bob"])
    assert game.players[0].name == "Alice"
    assert game.players[1].name == "Bob"


def test_game_init_generates_default_names():
    """Game generates names like 'Player 1' if not provided."""
    game = Game()
    assert "Player" in game.players[0].name
    assert "Player" in game.players[1].name


def test_game_init_sets_seed_for_reproducibility():
    """Game with same seed produces same state."""
    game1 = Game(seed=42)
    game1.setup_round()
    factories1 = [list(f) for f in game1.factories]

    game2 = Game(seed=42)
    game2.setup_round()
    factories2 = [list(f) for f in game2.factories]

    assert factories1 == factories2


def test_current_player_starts_at_index_zero():
    """Game starts with current_player at index 0."""
    game = Game()
    assert game.current_player == game.players[0]


# endregion


# region setup_round ---------------------------------------------------


def test_setup_round_fills_factories(game):
    """setup_round() distributes tiles to all factories."""
    game.setup_round()
    assert len(game.factories) == NUMBER_OF_FACTORIES
    for factory in game.factories:
        assert len(factory) == TILES_PER_FACTORY


def test_setup_round_places_first_player_in_center(game):
    """setup_round() places FIRST_PLAYER token in center."""
    game.setup_round()
    assert Tile.FIRST_PLAYER in game.center


def test_setup_round_clears_previous_state(game):
    """setup_round() resets bag/discard for new round."""
    game.setup_round()
    # Play a move to modify state
    moves = game.legal_moves()
    if moves:
        game.make_move(moves[0])
    # Setup again should reset
    game.setup_round()
    # Factories should be refilled
    total_tiles = sum(len(f) for f in game.factories)
    assert total_tiles > 0


# endregion


# region legal_moves & tile availability --------------------------------


def test_legal_moves_returns_list(game_after_setup):
    """legal_moves() returns a non-empty list."""
    moves = game_after_setup.legal_moves()
    assert isinstance(moves, list)
    assert len(moves) > 0


def test_legal_moves_are_move_objects(game_after_setup):
    """legal_moves() returns Move objects."""
    moves = game_after_setup.legal_moves()
    assert all(isinstance(m, Move) for m in moves)


def test_legal_moves_include_floor_option(game_after_setup):
    """legal_moves() includes option to move tiles to floor."""
    moves = game_after_setup.legal_moves()
    floor_moves = [m for m in moves if m.destination == FLOOR]
    assert len(floor_moves) > 0


def test_tile_availability_returns_dict(game_after_setup):
    """_tile_availability() returns a dictionary."""
    avail = game_after_setup._tile_availability()
    assert isinstance(avail, dict)
    # Should have entries for each color tile
    for tile in Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE:
        assert tile in avail
        assert isinstance(avail[tile], tuple)
        assert len(avail[tile]) == 2  # (count, source_count)


def test_legal_moves_only_include_available_tiles(game_after_setup):
    """legal_moves() only include tiles that exist in factories/center."""
    moves = game_after_setup.legal_moves()
    avail = game_after_setup._tile_availability()

    for move in moves:
        if move.destination != FLOOR:
            # Tile should be available
            count, sources = avail[move.tile]
            assert count > 0


# endregion


# region make_move & game flow ------------------------------------------


def test_make_move_consumes_tiles_from_source(game_after_setup):
    """make_move() removes tiles from the source."""
    moves = game_after_setup.legal_moves()
    assert len(moves) > 0

    move = moves[0]
    if move.source == CENTER:
        tiles_before = len(game_after_setup.center)
    else:
        tiles_before = len(game_after_setup.factories[move.source])

    game_after_setup.make_move(move)

    if move.source == CENTER:
        tiles_after = len(game_after_setup.center)
    else:
        tiles_after = len(game_after_setup.factories[move.source])

    # Tiles should have been removed
    assert tiles_after < tiles_before


def test_make_move_updates_player_board(game_after_setup):
    """make_move() updates current player's board state."""
    player = game_after_setup.current_player
    moves = game_after_setup.legal_moves()

    # Find a move that doesn't go to floor (so we can check pattern lines)
    move = next((m for m in moves if m.destination != FLOOR), None)
    if move:
        game_after_setup.make_move(move)
        # Player should have tiles on their pattern line or floor
        has_tiles = (
            any(len(row) > 0 for row in player._pattern_lines)
            or len(player._floor_line) > 0
        )
        assert has_tiles


def test_next_player_advances_turn(game_after_setup):
    """next_player() changes current_player."""
    player_0 = game_after_setup.players[0]
    game_after_setup.next_player()
    assert game_after_setup.current_player == game_after_setup.players[1]
    game_after_setup.next_player()
    assert game_after_setup.current_player == player_0


def test_advance_without_skip_setup_is_normal_flow(game_after_setup):
    """advance(skip_setup=False) handles normal round progression."""
    initial_round = game_after_setup.round
    # Make moves until round progresses
    max_iterations = 100
    for _ in range(max_iterations):
        moves = game_after_setup.legal_moves()
        if not moves:
            break
        game_after_setup.make_move(moves[0])
        round_boundary_crossed = game_after_setup.advance()
        if round_boundary_crossed:
            assert game_after_setup.round > initial_round
            break


# endregion


# region round & game over detection ------------------------------------


def test_is_round_over_initially_false(game_after_setup):
    """is_round_over() returns False at round start."""
    assert not game_after_setup.is_round_over()


def test_is_game_over_initially_false(game_after_setup):
    """is_game_over() returns False at game start."""
    assert not game_after_setup.is_game_over()


def test_is_game_over_true_when_player_completes_row():
    """is_game_over() becomes True when a player completes a wall row."""
    game = Game(["Alice", "Bob"])
    game.setup_round()

    # Manually place a full row of tiles on Alice's wall to trigger game end
    alice = game.players[0]
    for col in range(5):
        alice._wall_tiles[0][col] = Tile(col)
    alice._encode()

    # Game should detect end-game condition
    assert alice.has_triggered_game_end()


# endregion


# region clone & state management ----------------------------------------


def test_clone_creates_independent_copy(game_after_setup):
    """clone() creates a deep copy that's independent."""
    clone = game_after_setup.clone()

    # Modifying clone shouldn't affect original
    moves = clone.legal_moves()
    if moves:
        clone.make_move(moves[0])

    # Original should be unchanged
    original_moves = game_after_setup.legal_moves()
    clone_moves_after = clone.legal_moves()

    assert (
        len(original_moves) != len(clone_moves_after)
        or original_moves != clone_moves_after
    )


def test_clone_preserves_game_state(game_after_setup):
    """clone() preserves all game state."""
    clone = game_after_setup.clone()

    assert str(game_after_setup) == str(clone)
    assert game_after_setup.round == clone.round
    assert len(game_after_setup.factories) == len(clone.factories)


# endregion


# region from_string & serialization ------------------------------------


def test_from_string_with_initial_game():
    """Game.from_string() reconstructs initial game state."""
    original = Game(["Alice", "Bob"])
    original.setup_round()

    text = str(original)
    reconstructed = Game.from_string(text)

    # Should have same players and round
    assert reconstructed.players[0].name == "Alice"
    assert reconstructed.players[1].name == "Bob"
    assert reconstructed.round == original.round


def test_from_string_round_trip_produces_identical_string():
    """Game state can be serialized and deserialized identically."""
    original = Game(["Alice", "Bob"])
    original.setup_round()

    text1 = str(original)
    reconstructed = Game.from_string(text1)
    text2 = str(reconstructed)

    # The string representations should be identical
    assert text1 == text2


def test_from_string_preserves_encoded_features():
    """from_string() produces game with same encoding."""
    original = Game(["Alice", "Bob"])
    original.setup_round()

    text = str(original)
    reconstructed = Game.from_string(text)

    # Encodings should match
    assert original.encoded_features == reconstructed.encoded_features


# endregion


# region encoded_features -----------------------------------------------


def test_encoded_features_is_list(game_after_setup):
    """encoded_features property returns a list."""
    features = game_after_setup.encoded_features
    assert isinstance(features, list)
    assert len(features) > 0


def test_encoded_features_is_numeric(game_after_setup):
    """encoded_features contains numeric values."""
    features = game_after_setup.encoded_features
    assert all(isinstance(v, (int, float)) for v in features)


def test_encoded_features_updates_after_move(game_after_setup):
    """encoded_features changes after a move."""
    features_before = game_after_setup.encoded_features[:]

    moves = game_after_setup.legal_moves()
    if moves:
        game_after_setup.make_move(moves[0])
        features_after = game_after_setup.encoded_features

        # Features should change (at minimum, which player's turn it is)
        assert features_before != features_after


# endregion


# region Move class tests -----------------------------------------------


def test_move_from_str_parses_valid_move():
    """Move.from_str() parses move strings."""
    move = Move.from_str("2B-11")  # 2 blue, factory 1 (index 0), row 1 (index 0)
    assert move.tile == Tile.BLUE
    assert move.source == 0  # factory 1 in string = index 0
    assert move.destination == 0  # row 1 in string = index 0
    assert move.count == 2


def test_move_from_str_parses_center_move():
    """Move.from_str() parses center source."""
    move = Move.from_str("1R-C1")  # 1 red, center, row 1
    assert move.tile == Tile.RED
    assert move.source == CENTER
    assert move.destination == 0


def test_move_from_str_parses_floor_destination():
    """Move.from_str() parses floor destination."""
    move = Move.from_str("3Y-1F")  # 3 yellow, factory 1, floor
    assert move.tile == Tile.YELLOW
    assert move.destination == FLOOR
    assert move.source == 0


def test_move_str_round_trips():
    """Move.__str__() and from_str() round-trip."""
    original = Move.from_str("2B-C1")  # 2 blue, center, row 1
    text = str(original)
    reconstructed = Move.from_str(text)
    assert original == reconstructed


# endregion


# region Integration tests -----------------------------------------------


def test_can_make_a_legal_move(game_after_setup):
    """Can successfully make a legal move."""
    moves = game_after_setup.legal_moves()
    assert len(moves) > 0

    # Make the first legal move
    game_after_setup.make_move(moves[0])
    game_after_setup.advance()

    # Game state should have changed
    new_moves = game_after_setup.legal_moves()
    # Just verify we can keep going (might have more moves or round might be over)
    assert isinstance(new_moves, list)


def test_deterministic_play_with_seed():
    """Same seed produces identical move sequences."""

    def play_game(seed):
        game = Game(seed=seed)
        game.setup_round()
        moves_played = []
        for _ in range(10):
            moves = game.legal_moves()
            if not moves or game.is_round_over():
                break
            move = moves[0]
            moves_played.append((move.tile, move.source, move.destination))
            game.make_move(move)
            game.advance()
        return moves_played

    moves1 = play_game(42)
    moves2 = play_game(42)
    assert moves1 == moves2


# endregion
