from __future__ import annotations

from enum import Enum

BOARD_SIZE = 8


class PieceType(Enum):
    KING = "K"
    QUEEN = "Q"
    BISHOP = "B"
    ROOK = "R"
    KNIGHT = "N"
    PAWN = "P"


class Player(Enum):
    WHITE = "W"
    BLACK = "B"


# Unicode glyphs indexed by (player, piece type), used for the text board.
_GLYPHS: dict[tuple[Player, PieceType], str] = {
    (Player.WHITE, PieceType.KING): "♔",
    (Player.WHITE, PieceType.QUEEN): "♕",
    (Player.WHITE, PieceType.BISHOP): "♗",
    (Player.WHITE, PieceType.ROOK): "♖",
    (Player.WHITE, PieceType.KNIGHT): "♘",
    (Player.WHITE, PieceType.PAWN): "♙",
    (Player.BLACK, PieceType.KING): "♚",
    (Player.BLACK, PieceType.QUEEN): "♛",
    (Player.BLACK, PieceType.BISHOP): "♝",
    (Player.BLACK, PieceType.ROOK): "♜",
    (Player.BLACK, PieceType.KNIGHT): "♞",
    (Player.BLACK, PieceType.PAWN): "♟",
}


class Piece:
    __slots__ = ("player", "type", "has_moved")

    def __init__(self, player: Player, type: PieceType):
        self.player = player
        self.type = type
        self.has_moved = False

    def __repr__(self) -> str:
        return _GLYPHS[(self.player, self.type)]
