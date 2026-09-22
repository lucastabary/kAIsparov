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

    @property
    def opponent(self) -> Player:
        return Player.BLACK if self is Player.WHITE else Player.WHITE


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


# Dense index for a (player, type) pair, in enum declaration order (so the piece-type
# half matches :data:`kaisparov.core.bitboard.PT_IDX`). Hot loops key flat tables with
# this int instead of hashing enum members, which costs an order of magnitude more.
PIECE_CODES: dict[tuple[Player, PieceType], int] = {
    (player, piece_type): player_index * len(PieceType) + type_index
    for player_index, player in enumerate(Player)
    for type_index, piece_type in enumerate(PieceType)
}
NUM_PIECE_CODES = len(PIECE_CODES)


class Piece:
    __slots__ = ("player", "type", "has_moved", "code")

    def __init__(self, player: Player, type: PieceType):
        self.player = player
        self.type = type
        self.has_moved = False
        self.code = PIECE_CODES[(player, type)]

    def __repr__(self) -> str:
        return _GLYPHS[(self.player, self.type)]
