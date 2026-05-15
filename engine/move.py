# engine/move.py
"""Move representation for Azul game actions."""

from dataclasses import dataclass

from engine.constants import (
    CENTER,
    CHAR_FOR_TILE,
    FLOOR,
    MOVE_DEST_FLOOR,
    MOVE_MARKER_FIRST_PLAYER,
    MOVE_MARKER_NORMAL,
    MOVE_MARKER_UNKNOWN,
    MOVE_SOURCE_CENTER,
    TILE_FOR_CHAR,
    Tile,
)


@dataclass
class Move:
    """A single game action: take all tiles of one color from a source and
    place them on a destination pattern line or the floor.

    Attributes:
        source:      Factory index (0..N-1) or CENTER (-1).
        tile:        The tile color being taken.
        destination: Pattern line index (0..4) or FLOOR (-2).
        count:       Number of color tiles taken (excludes FIRST_PLAYER).
        took_first:  True if FIRST_PLAYER was also taken from the center.

    Compact string format: {count}{tile}{marker}{source}{destination}
        count:       number of color tiles
        tile:        single char from TILE_CHAR
        marker:      - normally, + if FIRST_PLAYER was taken
        source:      C for center, 1-5 for factory (1-based)
        destination: F for floor, 1-5 for pattern line row (1-based)

    Examples:
        2W-C3 — 2 white from center to row 3
        2W+C3 — same but also took the first-player tile
        1B-2F — 1 blue from factory 2 to floor
        4R+1F — 4 red from factory 1 to floor, took first-player tile
    """

    tile: Tile
    source: int
    destination: int
    count: int = 0
    took_first: bool = False

    def __str__(self) -> str:
        marker = (
            MOVE_MARKER_UNKNOWN
            if self.count == 0
            else MOVE_MARKER_FIRST_PLAYER if self.took_first else MOVE_MARKER_NORMAL
        )
        source_str = (
            MOVE_SOURCE_CENTER if self.source == CENTER else str(self.source + 1)
        )
        dest_str = (
            MOVE_DEST_FLOOR if self.destination == FLOOR else str(self.destination + 1)
        )
        return f"{self.count}{CHAR_FOR_TILE[self.tile]}{marker}{source_str}{dest_str}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return NotImplemented
        return (
            self.tile == other.tile
            and self.source == other.source
            and self.destination == other.destination
        )

    def __hash__(self) -> int:
        return hash((self.tile, self.source, self.destination))

    @classmethod
    def from_str(cls, s: str) -> "Move":
        """Parse a compact move string back into a Move.

        Format: {count}{tile}{marker}{source}{destination}
        count is 1 digit normally, 2 digits when >= 10.

        Raises ValueError if the string is not a valid move format.

        Examples:
            Move.from_str("2W-C3")  # 2 white, center, row 3, no first player
            Move.from_str("4R+1F")  # 4 red, factory 1, floor, took first player
            Move.from_str("10R+CF") # 10 red, center, floor, took first player
        """
        try:
            count = int(s[:-4])
            tile = TILE_FOR_CHAR[s[-4]]
            if tile is None:
                raise ValueError(f"invalid tile char: {s[-4]!r}")
            took_first = s[-3] == MOVE_MARKER_FIRST_PLAYER
            source = CENTER if s[-2] == MOVE_SOURCE_CENTER else int(s[-2]) - 1
            destination = FLOOR if s[-1] == MOVE_DEST_FLOOR else int(s[-1]) - 1
            return cls(
                tile=tile,
                source=source,
                destination=destination,
                count=count,
                took_first=took_first,
            )
        except (IndexError, KeyError) as exc:
            raise ValueError(f"invalid move string: {s!r}") from exc
