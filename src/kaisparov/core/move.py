"""A move: where from, where to, and what a promoting pawn becomes.

Historically a move was a bare ``(source, dest)`` pair, because the capture-the-king
variant had no promotion. Standard chess does, and ``e7e8`` is four different moves,
so the pair grew a third field.

:class:`Move` is a :class:`~typing.NamedTuple`, hence still a tuple: ``game.make(*move)``
and ``move[0]`` keep working, and ``promotion`` defaults to ``None`` so a normal move
is written ``Move(source, dest)``. Only code that unpacks a move into exactly two names
(``source, dest = move``) had to change.

The conversions to and from :class:`chess.Move` live here so the rest of the package
speaks ``(col, row)`` coordinates and never touches python-chess square indices.
"""

from __future__ import annotations

from typing import NamedTuple

import chess

from kaisparov.core.coords import Coord
from kaisparov.core.pieces import PieceType

# python-chess piece types are ints; ours is an enum. The board index conventions
# already agree — ``chess.square(file, rank) == rank * 8 + file == coord_to_index``.
TO_CHESS_PIECE: dict[PieceType, int] = {
    PieceType.PAWN: chess.PAWN,
    PieceType.KNIGHT: chess.KNIGHT,
    PieceType.BISHOP: chess.BISHOP,
    PieceType.ROOK: chess.ROOK,
    PieceType.QUEEN: chess.QUEEN,
    PieceType.KING: chess.KING,
}
FROM_CHESS_PIECE: dict[int, PieceType] = {v: k for k, v in TO_CHESS_PIECE.items()}

# What a pawn may become. Queen first: it is the right answer in almost every
# position, so a caller that wants "the" promotion can take ``PROMOTION_PIECES[0]``.
PROMOTION_PIECES: tuple[PieceType, ...] = (
    PieceType.QUEEN,
    PieceType.ROOK,
    PieceType.BISHOP,
    PieceType.KNIGHT,
)


def coord_to_square(coord: Coord) -> int:
    """``(col, row)`` -> python-chess square index."""
    return coord[1] * 8 + coord[0]


def square_to_coord(square: int) -> Coord:
    """python-chess square index -> ``(col, row)``."""
    return (square & 7, square >> 3)


class Move(NamedTuple):
    """``source -> dest``, plus the piece a promoting pawn becomes."""

    source: Coord
    dest: Coord
    promotion: PieceType | None = None

    @classmethod
    def from_chess(cls, move: chess.Move) -> Move:
        return cls(
            square_to_coord(move.from_square),
            square_to_coord(move.to_square),
            FROM_CHESS_PIECE[move.promotion] if move.promotion else None,
        )

    def to_chess(self) -> chess.Move:
        return chess.Move(
            coord_to_square(self.source),
            coord_to_square(self.dest),
            promotion=TO_CHESS_PIECE[self.promotion] if self.promotion else None,
        )

    def uci(self) -> str:
        return self.to_chess().uci()

    @classmethod
    def from_uci(cls, text: str) -> Move:
        return cls.from_chess(chess.Move.from_uci(text))

    def __str__(self) -> str:
        return self.uci()


__all__ = [
    "Move",
    "PROMOTION_PIECES",
    "TO_CHESS_PIECE",
    "FROM_CHESS_PIECE",
    "coord_to_square",
    "square_to_coord",
]
