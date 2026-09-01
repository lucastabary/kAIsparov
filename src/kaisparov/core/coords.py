"""Single source of truth for board coordinates.

A coordinate is a ``(col, row)`` == ``(x, y)`` tuple with the origin at the
bottom-left; ``x`` is the file (0..7) and ``y`` is the rank (0..7). White pawns
advance towards higher ``y``. The linear index used by the graph models is
``row * BOARD_SIZE + col`` (row-major).
"""

from __future__ import annotations

from collections.abc import Iterator

from kaisparov.core.pieces import BOARD_SIZE, Player

Coord = tuple[int, int]


def in_bounds(coord: Coord) -> bool:
    x, y = coord
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def coord_to_index(coord: Coord) -> int:
    x, y = coord
    return y * BOARD_SIZE + x


def index_to_coord(index: int) -> Coord:
    return (index % BOARD_SIZE, index // BOARD_SIZE)


def all_squares() -> Iterator[Coord]:
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            yield (x, y)


def to_pov_coord(coord: Coord, player: Player) -> Coord:
    """Map a real coordinate into ``player``'s point of view.

    White sees the board unchanged; Black's rank axis is flipped so its pieces
    appear at the bottom. The transform is its own inverse.
    """
    if player == Player.WHITE:
        return coord
    x, y = coord
    return (x, BOARD_SIZE - 1 - y)


# The POV transform is an involution: applying it twice is the identity.
from_pov_coord = to_pov_coord
