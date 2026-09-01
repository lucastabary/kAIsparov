"""Chess-domain helpers.

Coordinate conversions live in :mod:`kaisparov.core.coords` (the single source of
truth) and are re-exported here for convenience / backward compatibility.
"""

from kaisparov.core.coords import coord_to_index, index_to_coord
from kaisparov.core.pieces import PieceType

_PIECE_VALUES: dict[PieceType, float] = {
    PieceType.PAWN: 0.01,
    PieceType.KNIGHT: 0.03,
    PieceType.BISHOP: 0.03,
    PieceType.ROOK: 0.05,
    PieceType.QUEEN: 0.09,
    PieceType.KING: 1.0,
}


def get_piece_value(piece_type: PieceType) -> float:
    """Return the material value of a piece type."""
    return _PIECE_VALUES.get(piece_type, 0.0)


__all__ = ["coord_to_index", "index_to_coord", "get_piece_value"]
