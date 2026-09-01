from __future__ import annotations

from dataclasses import dataclass

from kaisparov.core import coords, movegen, rules
from kaisparov.core.coords import Coord
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player

Grid = list[list["Piece | None"]]


@dataclass
class Undo:
    """Everything needed to reverse a single ``make`` in O(1)."""

    source: Coord
    dest: Coord
    piece: Piece
    captured: Piece | None
    piece_had_moved: bool
    castle: tuple[Piece, Coord, Coord, bool] | None
    prev_turn: Player
    turn_advanced: bool


class ChessGame:
    """Mutable board state with fast, reversible moves.

    Coordinates are ``(col, row)``; ``grid[col][row]`` holds a ``Piece`` or
    ``None``. The variant ends when a king is captured (see :mod:`kaisparov.core`).
    """

    def __init__(self, initial_board: Grid | None = None, turn: Player = Player.WHITE):
        self.grid: Grid = (
            self._build_grid() if initial_board is None else self._clone_grid(initial_board)
        )
        self.turn: Player = turn
        self.count = 0

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _empty_grid() -> Grid:
        return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    @classmethod
    def _clone_grid(cls, grid: Grid) -> Grid:
        if len(grid) != BOARD_SIZE or any(len(col) != BOARD_SIZE for col in grid):
            raise ValueError(f"Board must be {BOARD_SIZE}x{BOARD_SIZE}")

        clone = cls._empty_grid()
        for x, y in coords.all_squares():
            piece = grid[x][y]
            if piece is None:
                continue
            copied = Piece(piece.player, piece.type)
            copied.has_moved = piece.has_moved
            clone[x][y] = copied
        return clone

    def _build_grid(self) -> Grid:
        """Return the standard starting position (grid[0][0] is bottom-left)."""
        grid = self._empty_grid()

        for x in range(BOARD_SIZE):
            grid[x][1] = Piece(Player.WHITE, PieceType.PAWN)
            grid[x][BOARD_SIZE - 2] = Piece(Player.BLACK, PieceType.PAWN)

        back_rank = [
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.BISHOP,
            PieceType.QUEEN,
            PieceType.KING,
            PieceType.BISHOP,
            PieceType.KNIGHT,
            PieceType.ROOK,
        ]
        for x, piece_type in enumerate(back_rank):
            grid[x][0] = Piece(Player.WHITE, piece_type)
            grid[x][BOARD_SIZE - 1] = Piece(Player.BLACK, piece_type)

        return grid

    def set_board(self, board: Grid, turn: Player | None = None) -> None:
        self.grid = self._clone_grid(board)
        if turn is not None:
            self.turn = turn
        self.count = 0

    def copy(self) -> ChessGame:
        """Return a deep, independent copy of the current state."""
        clone = ChessGame(initial_board=self.grid, turn=self.turn)
        clone.count = self.count
        return clone

    # ------------------------------------------------------------------ moves
    def possible_moves(self, source: Coord) -> list[Coord]:
        """Pseudo-legal destinations for the piece on ``source`` (empty if none)."""
        if not coords.in_bounds(source):
            return []
        return movegen.pseudo_legal_moves(self.grid, source)

    def is_move_valid(self, source: Coord, dest: Coord) -> bool:
        if not coords.in_bounds(source) or not coords.in_bounds(dest):
            return False
        piece = self.grid[source[0]][source[1]]
        if piece is None or piece.player != self.turn:
            return False
        return dest in self.possible_moves(source)

    def make(self, source: Coord, dest: Coord) -> Undo:
        """Apply a move assumed legal, returning an :class:`Undo` handle.

        Callers that cannot guarantee legality should use :meth:`play`.
        """
        sx, sy = source
        dx, dy = dest
        piece = self.grid[sx][sy]
        assert piece is not None, f"no piece to move on {source}"

        captured = self.grid[dx][dy]
        piece_had_moved = piece.has_moved

        self.grid[dx][dy] = piece
        self.grid[sx][sy] = None
        piece.has_moved = True

        castle: tuple[Piece, Coord, Coord, bool] | None = None
        if piece.type == PieceType.KING and abs(dx - sx) == 2:
            if dx > sx:  # kingside
                rook_src, rook_dest = (BOARD_SIZE - 1, dy), (sx + 1, dy)
            else:  # queenside
                rook_src, rook_dest = (0, dy), (sx - 1, dy)
            rook = self.grid[rook_src[0]][rook_src[1]]
            if rook is not None and rook.type == PieceType.ROOK and rook.player == piece.player:
                castle = (rook, rook_src, rook_dest, rook.has_moved)
                self.grid[rook_dest[0]][rook_dest[1]] = rook
                self.grid[rook_src[0]][rook_src[1]] = None
                rook.has_moved = True

        self.count += 1
        prev_turn = self.turn
        king_captured = captured is not None and captured.type == PieceType.KING
        turn_advanced = not king_captured
        if turn_advanced:
            self.turn = Player.BLACK if self.turn == Player.WHITE else Player.WHITE

        return Undo(
            source, dest, piece, captured, piece_had_moved, castle, prev_turn, turn_advanced
        )

    def unmake(self, undo: Undo) -> None:
        """Reverse a move produced by :meth:`make`."""
        sx, sy = undo.source
        dx, dy = undo.dest

        self.grid[sx][sy] = undo.piece
        self.grid[dx][dy] = undo.captured
        undo.piece.has_moved = undo.piece_had_moved

        if undo.castle is not None:
            rook, rook_src, rook_dest, rook_had_moved = undo.castle
            self.grid[rook_src[0]][rook_src[1]] = rook
            self.grid[rook_dest[0]][rook_dest[1]] = None
            rook.has_moved = rook_had_moved

        self.count -= 1
        self.turn = undo.prev_turn

    def play(self, source: Coord, dest: Coord) -> Piece | None:
        """Validate then apply a move. Returns the captured piece, or ``None``.

        Returns ``None`` both for an illegal move and for a legal non-capturing
        move; callers that need to distinguish should check :meth:`is_move_valid`.
        """
        if not self.is_move_valid(source, dest):
            return None
        return self.make(source, dest).captured

    # -------------------------------------------------------------- viewpoints
    def to_pov_coord(self, coord: Coord) -> Coord:
        return coords.to_pov_coord(coord, self.turn)

    def from_pov_coord(self, pov_coord: Coord) -> Coord:
        return coords.from_pov_coord(pov_coord, self.turn)

    def get_pov_grid(self) -> Grid:
        """Return the grid from the current player's point of view."""
        if self.turn == Player.WHITE:
            return self.grid
        return [
            [self.grid[col][BOARD_SIZE - 1 - row] for row in range(BOARD_SIZE)]
            for col in range(BOARD_SIZE)
        ]

    def is_in_check(self, player: Player) -> bool:
        return rules.is_in_check(self.grid, player)

    # -------------------------------------------------------------------- misc
    def print_grid(self) -> None:
        print(f"turn: {self.turn.name} | count: {self.count}")
        for row in range(BOARD_SIZE - 1, -1, -1):
            print(f"{row} ", end="")
            for col in range(BOARD_SIZE):
                cell = self.grid[col][row]
                print("  " if cell is None else str(cell), end=" ")
            print()
        print("  ", end="")
        for col in range(BOARD_SIZE):
            print(f"{col}  ", end="")
        print()
