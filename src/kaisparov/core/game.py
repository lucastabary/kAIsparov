"""The board: standard chess, backed by python-chess.

:class:`ChessGame` is a thin, mutable facade over :class:`chess.Board`. It keeps the
API the rest of the package already speaks — ``(col, row)`` coordinates, ``Piece``
objects, ``make``/``unmake``, ``possible_moves`` — so python-chess stays an
implementation detail of this module and never leaks into the models, the agents or
the benchmark.

What it buys, over the hand-written engine it replaces: legal move generation with
real pin detection, promotion (all four pieces), checkmate, stalemate, the fifty-move
rule, threefold repetition and insufficient material — correct, and measurably faster
than filtering pseudo-legal moves with a make/unmake per candidate.

The one thing to know about performance: :attr:`grid` rebuilds a 8x8 list of
:class:`Piece` objects and is *cached*, invalidated on every ``make``/``unmake``. It
exists for the UI, the heuristics and the tests. Hot paths that only need occupancy
should read :attr:`board` and its bitboards directly (``board.pawns &
board.occupied_co[chess.WHITE]``), which are already in the ``row * 8 + col``
convention this project uses — see :mod:`kaisparov.models.rgcn.processor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.polyglot

from kaisparov.core import coords, draw
from kaisparov.core.coords import Coord
from kaisparov.core.move import (
    FROM_CHESS_PIECE,
    TO_CHESS_PIECE,
    Move,
    coord_to_square,
    square_to_coord,
)
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player

Grid = list[list["Piece | None"]]

# Castling rights, as (colour, king square, rook square, right) — used both to read
# rights off a hand-built grid and to derive a piece's ``has_moved`` flag from them.
_CASTLING_SLOTS = (
    (Player.WHITE, chess.E1, chess.H1, chess.BB_H1),
    (Player.WHITE, chess.E1, chess.A1, chess.BB_A1),
    (Player.BLACK, chess.E8, chess.H8, chess.BB_H8),
    (Player.BLACK, chess.E8, chess.A8, chess.BB_A8),
)


@dataclass
class Undo:
    """Everything needed to reverse a single :meth:`ChessGame.make`.

    The position itself is restored by :meth:`chess.Board.pop`; this carries the
    derived state the facade keeps on the side, plus the captured piece, which
    callers use for material reward and move review.
    """

    move: Move
    piece: Piece
    captured: Piece | None
    captured_square: Coord | None  # differs from dest on en passant
    is_castle: bool
    is_en_passant: bool
    prev_turn: Player
    prev_last_move: tuple[Coord, Coord] | None
    prev_zobrist: int


class ChessGame:
    """Mutable standard-chess position with fast, reversible moves.

    Coordinates are ``(col, row)`` with the origin bottom-left; ``grid[col][row]``
    holds a :class:`Piece` or ``None``. White advances toward higher ``row``.
    """

    def __init__(
        self,
        initial_board: Grid | None = None,
        turn: Player = Player.WHITE,
        en_passant_target: Coord | None = None,
        *,
        board: chess.Board | None = None,
    ):
        if board is not None:
            self.board = board
        elif initial_board is None:
            self.board = chess.Board()
        else:
            self.board = _board_from_grid(initial_board, turn, en_passant_target)

        self.count = 0
        # (source, dest) of the last move played — for UI highlighting.
        self.last_move: tuple[Coord, Coord] | None = None
        self._grid: Grid | None = None
        self._zobrist: int | None = None
        self.position_history: list[int] = [self.zobrist]

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _empty_grid() -> Grid:
        return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    def set_board(self, board: Grid, turn: Player | None = None) -> None:
        """Replace the position with ``board`` (and optionally the side to move)."""
        self.board = _board_from_grid(board, turn if turn is not None else self.turn, None)
        self.count = 0
        self.last_move = None
        self._reset_derived()

    def _reset_derived(self) -> None:
        self._grid = None
        self._zobrist = None
        self.position_history = [self.zobrist]

    def _invalidate(self) -> None:
        self._grid = None
        self._zobrist = None

    def copy(self) -> ChessGame:
        """A deep copy that shares nothing mutable with this game."""
        clone = ChessGame(board=self.board.copy())
        clone.count = self.count
        clone.last_move = self.last_move
        clone.position_history = list(self.position_history)
        return clone

    # ------------------------------------------------------------------ state
    @property
    def grid(self) -> Grid:
        """The position as ``grid[col][row]``, rebuilt on demand and cached.

        The :class:`Piece` objects are fresh on every rebuild: mutating one does not
        change the position. Write through :meth:`make` or :meth:`set_board`.
        """
        if self._grid is None:
            self._grid = _grid_from_board(self.board)
        return self._grid

    @property
    def turn(self) -> Player:
        return Player.WHITE if self.board.turn else Player.BLACK

    @property
    def en_passant_target(self) -> Coord | None:
        """The square a pawn just skipped, if any (armed after every double push)."""
        ep = self.board.ep_square
        return None if ep is None else square_to_coord(ep)

    @property
    def halfmove_clock(self) -> int:
        """Plies since the last capture or pawn move — the fifty-move counter."""
        return self.board.halfmove_clock

    @property
    def zobrist(self) -> int:
        """A fingerprint of the position (pieces, side to move, castling, en passant)."""
        if self._zobrist is None:
            self._zobrist = chess.polyglot.zobrist_hash(self.board)
        return self._zobrist

    # ------------------------------------------------------------------ moves
    def legal_moves(self) -> list[Move]:
        """Every legal move for the side to move, promotions spelled out one by one."""
        return [Move.from_chess(m) for m in self.board.legal_moves]

    def possible_moves(self, source: Coord) -> list[Coord]:
        """Destination squares reachable from ``source``, each listed once.

        The four promotions of one pawn push share a destination, so they collapse to
        a single entry here. Callers that need the promotion piece want
        :meth:`legal_moves`.
        """
        from_square = coord_to_square(source)
        seen: dict[Coord, None] = {}
        for move in self.board.legal_moves:
            if move.from_square == from_square:
                seen[square_to_coord(move.to_square)] = None
        return list(seen)

    def is_move_valid(self, source: Coord, dest: Coord, promotion: PieceType | None = None) -> bool:
        """Whether this move is legal right now.

        ``promotion=None`` on a move that requires one is read as "any promotion",
        matching :meth:`possible_moves`; :meth:`make` then queens by default.
        """
        from_square, to_square = coord_to_square(source), coord_to_square(dest)
        wanted = TO_CHESS_PIECE[promotion] if promotion is not None else None
        for move in self.board.legal_moves:
            if move.from_square != from_square or move.to_square != to_square:
                continue
            if wanted is None or move.promotion == wanted:
                return True
        return False

    def _resolve(self, source: Coord, dest: Coord, promotion: PieceType | None) -> chess.Move:
        """Build the python-chess move, queening when a promotion is needed but unnamed."""
        move = chess.Move(
            coord_to_square(source),
            coord_to_square(dest),
            promotion=TO_CHESS_PIECE[promotion] if promotion is not None else None,
        )
        if move.promotion is None and _needs_promotion(self.board, move):
            move.promotion = chess.QUEEN
        return move

    def make(self, source: Coord, dest: Coord, promotion: PieceType | None = None) -> Undo:
        """Apply a move assumed legal, returning an :class:`Undo` handle.

        Callers that cannot guarantee legality should use :meth:`play`. A pawn move to
        the last rank without an explicit ``promotion`` becomes a queen.
        """
        board = self.board
        move = self._resolve(source, dest, promotion)

        moving = board.piece_at(move.from_square)
        assert moving is not None, f"no piece to move on {source}"
        piece = _piece_from_chess(moving, board, move.from_square)

        is_en_passant = board.is_en_passant(move)
        if is_en_passant:
            captured_square: Coord | None = (dest[0], source[1])
            captured = Piece(_other(piece.player), PieceType.PAWN)
        else:
            taken = board.piece_at(move.to_square)
            captured = None if taken is None else _piece_from_chess(taken, board, move.to_square)
            captured_square = dest if captured is not None else None

        undo = Undo(
            move=Move(source, dest, FROM_CHESS_PIECE[move.promotion] if move.promotion else None),
            piece=piece,
            captured=captured,
            captured_square=captured_square,
            is_castle=board.is_castling(move),
            is_en_passant=is_en_passant,
            prev_turn=self.turn,
            prev_last_move=self.last_move,
            prev_zobrist=self.zobrist,
        )

        board.push(move)
        self.count += 1
        self.last_move = (source, dest)
        self._invalidate()
        self.position_history.append(self.zobrist)
        return undo

    def unmake(self, undo: Undo) -> None:
        """Reverse a move produced by :meth:`make`."""
        self.board.pop()
        self.count -= 1
        self.last_move = undo.prev_last_move
        # make/unmake are perfectly nested (searches included), so the history always
        # comes back to what it was.
        self.position_history.pop()
        self._grid = None
        self._zobrist = undo.prev_zobrist

    def play(self, source: Coord, dest: Coord, promotion: PieceType | None = None) -> Piece | None:
        """Validate then apply a move. Returns the captured piece, or ``None``.

        Returns ``None`` both for an illegal move and for a legal non-capturing move;
        callers that need to distinguish should check :meth:`is_move_valid`.
        """
        if not self.is_move_valid(source, dest, promotion):
            return None
        return self.make(source, dest, promotion).captured

    # -------------------------------------------------------------------- pov
    def to_pov_coord(self, coord: Coord, player: Player | None = None) -> Coord:
        return coords.to_pov_coord(coord, player if player is not None else self.turn)

    def from_pov_coord(self, pov_coord: Coord, player: Player | None = None) -> Coord:
        return coords.from_pov_coord(pov_coord, player if player is not None else self.turn)

    def get_pov_grid(self, player: Player | None = None) -> Grid:
        """The grid as ``player`` sees it, own pieces at the bottom."""
        who = player if player is not None else self.turn
        grid = self.grid
        pov = self._empty_grid()
        for x, y in coords.all_squares():
            px, py = coords.to_pov_coord((x, y), who)
            pov[px][py] = grid[x][y]
        return pov

    # ----------------------------------------------------------- game outcome
    def is_in_check(self, player: Player) -> bool:
        """True if ``player``'s king is attacked."""
        king = self.board.king(player == Player.WHITE)
        if king is None:
            return False
        return self.board.is_attacked_by(player != Player.WHITE, king)

    def is_checkmate(self) -> bool:
        return self.board.is_checkmate()

    def is_stalemate(self) -> bool:
        return self.board.is_stalemate()

    # ------------------------------------------------------------------- draws
    def repetition_count(self) -> int:
        """How many times the current position has occurred in this game (>= 1)."""
        current = self.position_history[-1]
        return self.position_history.count(current)

    def draw_reason(self, draw_rules: draw.DrawRules | None = draw.DEFAULT_RULES) -> str | None:
        return draw.draw_reason(self, draw_rules)

    def is_draw(self, draw_rules: draw.DrawRules | None = draw.DEFAULT_RULES) -> bool:
        return draw.draw_reason(self, draw_rules) is not None

    # -------------------------------------------------------------------- misc
    def fen(self) -> str:
        return self.board.fen()

    def print_grid(self) -> None:
        """Print the board from White's side, ranks 8 down to 1."""
        grid = self.grid
        for y in range(BOARD_SIZE - 1, -1, -1):
            print(" ".join(str(grid[x][y]) if grid[x][y] else "." for x in range(BOARD_SIZE)))

    def __repr__(self) -> str:
        return f"ChessGame({self.board.fen()!r})"


# --------------------------------------------------------------------- helpers
def _other(player: Player) -> Player:
    return Player.BLACK if player == Player.WHITE else Player.WHITE


def _needs_promotion(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    rank = chess.square_rank(move.to_square)
    return rank == 7 if piece.color else rank == 0


def _has_moved(board: chess.Board, square: int, piece: chess.Piece) -> bool:
    """Best-effort ``has_moved``, derived from castling rights.

    python-chess tracks castling as a right on a rook square, not as a per-piece flag.
    Only kings and rooks ever needed the flag (it gated castling), so it is exact for
    them and conservatively ``True`` for everything else — nothing reads it otherwise.
    """
    if piece.piece_type not in (chess.KING, chess.ROOK):
        return True
    player = Player.WHITE if piece.color else Player.BLACK
    for owner, king_sq, rook_sq, right in _CASTLING_SLOTS:
        if owner != player or not board.castling_rights & right:
            continue
        if piece.piece_type == chess.KING and square == king_sq:
            return False
        if piece.piece_type == chess.ROOK and square == rook_sq:
            return False
    return True


def _piece_from_chess(piece: chess.Piece, board: chess.Board, square: int) -> Piece:
    out = Piece(Player.WHITE if piece.color else Player.BLACK, FROM_CHESS_PIECE[piece.piece_type])
    out.has_moved = _has_moved(board, square, piece)
    return out


def _grid_from_board(board: chess.Board) -> Grid:
    grid: Grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for square, piece in board.piece_map().items():
        col, row = square_to_coord(square)
        grid[col][row] = _piece_from_chess(piece, board, square)
    return grid


def _board_from_grid(grid: Grid, turn: Player, en_passant_target: Coord | None) -> chess.Board:
    """Build a :class:`chess.Board` from a hand-made grid (curriculum, benchmark, UI).

    Castling rights are read off the ``has_moved`` flags, the way the grid engine used
    to offer castling: a right survives only when both the king and that rook sit on
    their home squares unmoved.
    """
    if len(grid) != BOARD_SIZE or any(len(col) != BOARD_SIZE for col in grid):
        raise ValueError(f"Board must be {BOARD_SIZE}x{BOARD_SIZE}")

    board = chess.Board(None)  # empty, no castling rights, White to move
    for x, y in coords.all_squares():
        piece = grid[x][y]
        if piece is None:
            continue
        board.set_piece_at(
            coord_to_square((x, y)),
            chess.Piece(TO_CHESS_PIECE[piece.type], piece.player == Player.WHITE),
        )

    rights = chess.BB_EMPTY
    for player, king_sq, rook_sq, right in _CASTLING_SLOTS:
        if _unmoved(grid, king_sq, PieceType.KING, player) and _unmoved(
            grid, rook_sq, PieceType.ROOK, player
        ):
            rights |= right
    board.castling_rights = rights

    board.turn = turn == Player.WHITE
    if en_passant_target is not None:
        board.ep_square = coord_to_square(en_passant_target)
    return board


def _unmoved(grid: Grid, square: int, piece_type: PieceType, player: Player) -> bool:
    col, row = square_to_coord(square)
    piece = grid[col][row]
    return (
        piece is not None
        and piece.type == piece_type
        and piece.player == player
        and not piece.has_moved
    )


__all__ = ["ChessGame", "Undo", "Grid"]
