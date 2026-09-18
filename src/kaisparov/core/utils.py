"""Chess-domain helpers.

Coordinate conversions live in :mod:`kaisparov.core.coords` (the single source of
truth) and are re-exported here for convenience / backward compatibility.
"""

from kaisparov.core.coords import coord_to_index, index_to_coord
from kaisparov.core.pieces import PieceType

# Standard chess material values. The king cannot be captured, so its value is a
# sentinel that only ever shows up if something counts it by mistake — material
# scores exclude kings, and winning is the flat ``checkmate`` reward term (see
# :mod:`kaisparov.training.reward`), controllable independently of this number.
_PIECE_VALUES: dict[PieceType, float] = {
    PieceType.PAWN: 1.0,
    PieceType.KNIGHT: 3.0,
    PieceType.BISHOP: 3.0,
    PieceType.ROOK: 5.0,
    PieceType.QUEEN: 9.0,
    PieceType.KING: 100.0,
}


def get_piece_value(piece_type: PieceType) -> float:
    """Return the material value of a piece type."""
    return _PIECE_VALUES.get(piece_type, 0.0)


__all__ = ["coord_to_index", "index_to_coord", "get_piece_value"]
