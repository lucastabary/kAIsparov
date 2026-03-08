from enum import Enum

BOARD_SIZE = 8

class PieceType(Enum):
    NONE = 0
    KING = "K"
    QUEEN = "Q"
    BISHOP = "B"
    ROOK = "R"
    HORSE  = "H"
    PAWN = "P"

class Player(Enum):
    NONE = 0
    WHITE = "W"
    BLACK = "B"

class Piece():
    def __init__(self, player:Player, type: PieceType):
        self.player = player
        self.type = type
        self.hasMoved = False

    def __repr__(self):
        if (self.type == PieceType.NONE): return "  "
        if self.player == Player.WHITE:
            if self.type == PieceType.KING: return "♔"
            if self.type == PieceType.QUEEN: return "♕"
            if self.type == PieceType.BISHOP: return "♗"
            if self.type == PieceType.ROOK: return "♖"
            if self.type == PieceType.HORSE: return "♘"
            if self.type == PieceType.PAWN: return "♙"
        else:
            if self.type == PieceType.KING: return "♚"
            if self.type == PieceType.QUEEN: return "♛"
            if self.type == PieceType.BISHOP: return "♝"
            if self.type == PieceType.ROOK: return "♜"
            if self.type == PieceType.HORSE: return "♞"
            if self.type == PieceType.PAWN: return "♟"

BOARD_RANGE = list(range(1, BOARD_SIZE))

PIECES_MOVES = {
    PieceType.KING: [(0,-1), (0,1), (1,-1), (1,0), (1,1), (-1,-1), (-1,0), (-1,1)],
    PieceType.QUEEN: [
        [(i,0) for i in BOARD_RANGE],
        [(-i,0) for i in BOARD_RANGE],
        [(0,j) for j in BOARD_RANGE],
        [(0,-j) for j in BOARD_RANGE],
        [(i,i) for i in BOARD_RANGE],
        [(-i,-i) for i in BOARD_RANGE],
        [(i,-i) for i in BOARD_RANGE],
        [(-i,i) for i in BOARD_RANGE]],
    PieceType.BISHOP: [
        [(i,i) for i in BOARD_RANGE],
        [(-i,-i) for i in BOARD_RANGE],
        [(i,-i) for i in BOARD_RANGE],
        [(-i,i) for i in BOARD_RANGE]],
    PieceType.ROOK: [
        [(i,0) for i in BOARD_RANGE],
        [(-i,0) for i in BOARD_RANGE],
        [(0,j) for j in BOARD_RANGE],
        [(0,-j) for j in BOARD_RANGE]],
    PieceType.HORSE: [(2,-1), (2,1), (1, -2), (1,2), (-2,1), (-2,-1), (-1, -2), (-1,2)],
    PieceType.PAWN: [(0,1)]
}