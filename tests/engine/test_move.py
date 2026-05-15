"""Tests for Move class."""

import pytest
from engine.move import Move
from engine.constants import Tile, CENTER, FLOOR


# region Initialization -------------------------------------------------


def test_move_init_requires_tile_and_source_destination():
    """Move can be created with tile, source, destination."""
    move = Move(tile=Tile.BLUE, source=0, destination=2)
    assert move.tile == Tile.BLUE
    assert move.source == 0
    assert move.destination == 2


def test_move_init_count_defaults_to_zero():
    """Move.count defaults to 0 if not provided."""
    move = Move(tile=Tile.RED, source=CENTER, destination=1)
    assert move.count == 0


def test_move_init_took_first_defaults_to_false():
    """Move.took_first defaults to False if not provided."""
    move = Move(tile=Tile.WHITE, source=2, destination=FLOOR)
    assert move.took_first is False


def test_move_init_with_all_fields():
    """Move can be initialized with all fields."""
    move = Move(
        tile=Tile.YELLOW,
        source=1,
        destination=3,
        count=3,
        took_first=True,
    )
    assert move.tile == Tile.YELLOW
    assert move.source == 1
    assert move.destination == 3
    assert move.count == 3
    assert move.took_first is True


# endregion


# region __str__ --------------------------------------------------------


def test_move_str_basic():
    """Move.__str__() formats basic moves."""
    move = Move(tile=Tile.BLUE, source=0, destination=2, count=2)
    assert str(move) == "2B-13"  # destination 2 (0-indexed) = row 3 (1-indexed)


def test_move_str_with_first_player():
    """Move.__str__() includes + marker when took_first is True."""
    move = Move(
        tile=Tile.RED,
        source=1,
        destination=FLOOR,
        count=3,
        took_first=True,
    )
    assert str(move) == "3R+2F"


def test_move_str_from_center():
    """Move.__str__() uses 'C' for center source."""
    move = Move(tile=Tile.WHITE, source=CENTER, destination=0, count=1)
    assert str(move) == "1W-C1"


def test_move_str_to_floor():
    """Move.__str__() uses 'F' for floor destination."""
    move = Move(tile=Tile.YELLOW, source=3, destination=FLOOR, count=2)
    assert str(move) == "2Y-4F"


def test_move_str_zero_count():
    """Move.__str__() uses ? marker when count is 0."""
    move = Move(tile=Tile.BLACK, source=0, destination=1, count=0)
    assert str(move) == "0K?12"


def test_move_str_double_digit_count():
    """Move.__str__() handles counts >= 10."""
    move = Move(tile=Tile.BLUE, source=CENTER, destination=2, count=12)
    assert str(move) == "12B-C3"


def test_move_str_all_tile_colors():
    """Move.__str__() works for all tile colors."""
    for tile in [Tile.BLUE, Tile.YELLOW, Tile.RED, Tile.BLACK, Tile.WHITE]:
        move = Move(tile=tile, source=0, destination=1, count=1)
        # Should not raise an error
        assert len(str(move)) > 0


def test_move_str_all_sources():
    """Move.__str__() works for all source types."""
    # Factory sources 0-4
    for source in range(5):
        move = Move(tile=Tile.BLUE, source=source, destination=1, count=1)
        s = str(move)
        # Factory index should be 1-5 in string
        assert s[-2] == str(source + 1)

    # Center source
    move = Move(tile=Tile.BLUE, source=CENTER, destination=1, count=1)
    assert str(move)[-2] == "C"


def test_move_str_all_destinations():
    """Move.__str__() works for all destination types."""
    # Pattern line destinations 0-4
    for dest in range(5):
        move = Move(tile=Tile.BLUE, source=0, destination=dest, count=1)
        s = str(move)
        # Row index should be 1-5 in string
        assert s[-1] == str(dest + 1)

    # Floor destination
    move = Move(tile=Tile.BLUE, source=0, destination=FLOOR, count=1)
    assert str(move)[-1] == "F"


# endregion


# region __repr__ -------------------------------------------------------


def test_move_repr_equals_str():
    """Move.__repr__() returns same as __str__()."""
    move = Move(tile=Tile.RED, source=2, destination=3, count=2, took_first=True)
    assert repr(move) == str(move)


def test_move_repr_is_string():
    """Move.__repr__() returns a string."""
    move = Move(tile=Tile.YELLOW, source=CENTER, destination=FLOOR, count=1)
    assert isinstance(repr(move), str)


# endregion


# region __eq__ ---------------------------------------------------------


def test_move_equality_same_moves():
    """Two moves with same tile/source/destination are equal."""
    move1 = Move(tile=Tile.BLUE, source=0, destination=2, count=2)
    move2 = Move(tile=Tile.BLUE, source=0, destination=2, count=5)
    assert move1 == move2  # count and took_first don't affect equality


def test_move_equality_ignores_count():
    """Move equality ignores count field."""
    move1 = Move(tile=Tile.RED, source=1, destination=FLOOR, count=1)
    move2 = Move(tile=Tile.RED, source=1, destination=FLOOR, count=10)
    assert move1 == move2


def test_move_equality_ignores_took_first():
    """Move equality ignores took_first field."""
    move1 = Move(
        tile=Tile.WHITE,
        source=CENTER,
        destination=1,
        took_first=False,
    )
    move2 = Move(
        tile=Tile.WHITE,
        source=CENTER,
        destination=1,
        took_first=True,
    )
    assert move1 == move2


def test_move_inequality_different_tile():
    """Moves with different tiles are not equal."""
    move1 = Move(tile=Tile.BLUE, source=0, destination=1)
    move2 = Move(tile=Tile.RED, source=0, destination=1)
    assert move1 != move2


def test_move_inequality_different_source():
    """Moves with different sources are not equal."""
    move1 = Move(tile=Tile.BLUE, source=0, destination=1)
    move2 = Move(tile=Tile.BLUE, source=1, destination=1)
    assert move1 != move2


def test_move_inequality_different_destination():
    """Moves with different destinations are not equal."""
    move1 = Move(tile=Tile.BLUE, source=0, destination=1)
    move2 = Move(tile=Tile.BLUE, source=0, destination=2)
    assert move1 != move2


def test_move_equality_not_implemented_for_other_types():
    """Move.__eq__ returns NotImplemented for non-Move types."""
    move = Move(tile=Tile.BLUE, source=0, destination=1)
    assert (move == "2B-11") is False
    assert (move == 1) is False
    assert (move is None) is False


# endregion


# region __hash__ -------------------------------------------------------


def test_move_hash_same_for_equal_moves():
    """Equal moves have the same hash."""
    move1 = Move(tile=Tile.BLUE, source=0, destination=2, count=2)
    move2 = Move(tile=Tile.BLUE, source=0, destination=2, count=5)
    assert hash(move1) == hash(move2)


def test_move_hash_based_on_tile_source_destination():
    """Hash is based only on tile, source, destination."""
    # Same tile/source/destination but different count/took_first
    move1 = Move(
        tile=Tile.RED,
        source=1,
        destination=FLOOR,
        count=1,
        took_first=False,
    )
    move2 = Move(
        tile=Tile.RED,
        source=1,
        destination=FLOOR,
        count=10,
        took_first=True,
    )
    assert hash(move1) == hash(move2)


def test_move_hashable_in_set():
    """Moves can be used in sets."""
    move1 = Move(tile=Tile.BLUE, source=0, destination=1, count=2)
    move2 = Move(tile=Tile.BLUE, source=0, destination=1, count=5)  # Equal to move1
    move3 = Move(tile=Tile.RED, source=0, destination=1, count=2)

    s = {move1, move2, move3}
    # move1 and move2 are equal, so set should have 2 elements
    assert len(s) == 2


def test_move_hashable_as_dict_key():
    """Moves can be used as dictionary keys."""
    move1 = Move(tile=Tile.YELLOW, source=CENTER, destination=3, count=1)
    move2 = Move(tile=Tile.YELLOW, source=CENTER, destination=3, count=10)

    d = {move1: "first"}
    d[move2] = "second"  # Same move, should overwrite

    assert len(d) == 1
    assert d[move1] == "second"


# endregion


# region from_str -------------------------------------------------------


def test_move_from_str_basic():
    """Move.from_str() parses basic move strings."""
    move = Move.from_str("2B-12")
    assert move.tile == Tile.BLUE
    assert move.source == 0
    assert move.destination == 1
    assert move.count == 2
    assert move.took_first is False


def test_move_from_str_with_first_player():
    """Move.from_str() parses + marker for took_first."""
    move = Move.from_str("3R+2F")
    assert move.tile == Tile.RED
    assert move.source == 1
    assert move.destination == FLOOR
    assert move.count == 3
    assert move.took_first is True


def test_move_from_str_center_source():
    """Move.from_str() parses 'C' for center source."""
    move = Move.from_str("1W-C1")
    assert move.tile == Tile.WHITE
    assert move.source == CENTER
    assert move.destination == 0
    assert move.count == 1


def test_move_from_str_floor_destination():
    """Move.from_str() parses 'F' for floor destination."""
    move = Move.from_str("2Y-4F")
    assert move.tile == Tile.YELLOW
    assert move.source == 3
    assert move.destination == FLOOR
    assert move.count == 2


def test_move_from_str_double_digit_count():
    """Move.from_str() parses counts >= 10."""
    move = Move.from_str("12B-C3")
    assert move.count == 12
    assert move.tile == Tile.BLUE
    assert move.source == CENTER
    assert move.destination == 2


def test_move_from_str_zero_count():
    """Move.from_str() parses moves with count 0."""
    move = Move.from_str("0K?12")
    assert move.count == 0
    assert move.tile == Tile.BLACK
    assert move.took_first is False  # ? marker = not took_first
    assert move.source == 0
    assert move.destination == 1


def test_move_from_str_all_tile_colors():
    """Move.from_str() works for all tile colors."""
    tiles = [
        ("B", Tile.BLUE),
        ("Y", Tile.YELLOW),
        ("R", Tile.RED),
        ("K", Tile.BLACK),
        ("W", Tile.WHITE),
    ]
    for char, tile in tiles:
        move = Move.from_str(f"1{char}-11")
        assert move.tile == tile


def test_move_from_str_invalid_tile_raises():
    """Move.from_str() raises ValueError for invalid tile character."""
    with pytest.raises(ValueError, match="invalid move string"):
        Move.from_str("1X-11")


def test_move_from_str_malformed_raises():
    """Move.from_str() raises ValueError for malformed strings."""
    with pytest.raises(ValueError):
        Move.from_str("ABC")


def test_move_from_str_empty_raises():
    """Move.from_str() raises ValueError for empty string."""
    with pytest.raises(ValueError):
        Move.from_str("")


def test_move_from_str_invalid_count_raises():
    """Move.from_str() raises ValueError for non-numeric count."""
    with pytest.raises(ValueError):
        Move.from_str("XB-11")


# endregion


# region Round-trip serialization ------------------------------------


def test_move_round_trip_basic():
    """Move can be serialized and deserialized."""
    original = Move(tile=Tile.BLUE, source=0, destination=2, count=2)
    text = str(original)
    reconstructed = Move.from_str(text)
    assert original == reconstructed


def test_move_round_trip_with_first_player():
    """Move with took_first round-trips correctly."""
    original = Move(
        tile=Tile.RED,
        source=1,
        destination=FLOOR,
        count=3,
        took_first=True,
    )
    text = str(original)
    reconstructed = Move.from_str(text)
    assert original == reconstructed
    assert reconstructed.took_first is True


def test_move_round_trip_center_floor():
    """Move from center to floor round-trips."""
    original = Move(
        tile=Tile.WHITE,
        source=CENTER,
        destination=FLOOR,
        count=1,
        took_first=True,
    )
    text = str(original)
    reconstructed = Move.from_str(text)
    assert original == reconstructed


def test_move_round_trip_preserves_count():
    """Round-trip preserves count field (not checked by equality)."""
    original = Move(
        tile=Tile.YELLOW,
        source=2,
        destination=1,
        count=4,
        took_first=False,
    )
    text = str(original)
    reconstructed = Move.from_str(text)
    assert reconstructed.count == 4


def test_move_round_trip_all_sources():
    """Round-trip works for all sources."""
    for source in range(5):
        original = Move(
            tile=Tile.BLUE,
            source=source,
            destination=1,
            count=1,
        )
        reconstructed = Move.from_str(str(original))
        assert original == reconstructed

    original = Move(tile=Tile.BLUE, source=CENTER, destination=1, count=1)
    reconstructed = Move.from_str(str(original))
    assert original == reconstructed


def test_move_round_trip_all_destinations():
    """Round-trip works for all destinations."""
    for dest in range(5):
        original = Move(
            tile=Tile.BLUE,
            source=0,
            destination=dest,
            count=1,
        )
        reconstructed = Move.from_str(str(original))
        assert original == reconstructed

    original = Move(tile=Tile.BLUE, source=0, destination=FLOOR, count=1)
    reconstructed = Move.from_str(str(original))
    assert original == reconstructed


# endregion
