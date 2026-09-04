"""Chess-domain helpers.

Coordinate conversions live in :mod:`kaisparov.core.coords` (the single source of
truth) and are re-exported here for convenience / backward compatibility.
"""

from kaisparov.core.coords import coord_to_index, index_to_coord
from kaisparov.core.pieces import PieceType

# Standard chess material values. The king is a large sentinel so the greedy
# MaterialAgent always prefers capturing it (which wins the game); training reward
# does NOT scale a king capture by this value — it uses the flat ``king_capture``
# term instead (see :mod:`kaisparov.training.reward`), keeping the win reward
# controllable independently of this sentinel.
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
