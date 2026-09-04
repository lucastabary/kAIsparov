"""Bitboard move generation for the capture-the-king variant.

A position is held as twelve 64-bit integers (6 piece types x 2 colours) plus
per-colour occupancy and a mailbox. Bit ``i`` is the square ``i = row * 8 + col``
— the same linear index :mod:`kaisparov.core.coords` uses — so ``col = i & 7`` and
``row = i >> 3``.

This mirrors the grid engine's rules exactly (pseudo-legal moves, no check
filtering; castling only from the home square with king+rook unmoved; en passant;
capture-the-king; no promotion), and is validated against it by ``perft`` and a
move-set equivalence test. One documented scope limit: pawn double-push is gated on
the pawn's start rank rather than a per-pawn ``has_moved`` flag — equivalent for any
position reachable from the standard start (an unmoved pawn is always on its start
rank there), which is the domain of perft and normal play.

Slider attacks use the classical ray method: a precomputed ray from the square,
masked at the first blocker found with a bitscan.
"""

from __future__ import annotations

from kaisparov.core.pieces import Piece, PieceType, Player

# Piece-type -> index 0..5. KING is 0 so "captured a king" is ``code % 6 == 0``.
_PTYPES = (
    PieceType.KING,
    PieceType.QUEEN,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.KNIGHT,
    PieceType.PAWN,
)
PT_IDX = {pt: i for i, pt in enumerate(_PTYPES)}
KING_IDX = 0
QUEEN_IDX = 1
BISHOP_IDX = 2
ROOK_IDX = 3
KNIGHT_IDX = 4
PAWN_IDX = 5

WHITE, BLACK = 0, 1

# Move flags.
NORMAL, DOUBLE, EPCAP, CASTLE_K, CASTLE_Q = 0, 1, 2, 3, 4

# Castling-right bits.
WK, WQ, BK, BQ = 1, 2, 4, 8

# ------------------------------------------------------------------ attack tables
# Direction (dcol, drow) and whether the nearest blocker on the ray is the lowest
# set bit (True) or the highest (False) — i.e. whether the ray goes "up" in index.
_DIRS = [(0, 1), (1, 0), (1, 1), (-1, 1), (0, -1), (-1, 0), (1, -1), (-1, -1)]
_POS = [True, True, True, True, False, False, False, False]
_ROOK_DIRS = (0, 1, 4, 5)
_BISHOP_DIRS = (2, 3, 6, 7)
_QUEEN_DIRS = (0, 1, 2, 3, 4, 5, 6, 7)

_KNIGHT_OFF = [(2, -1), (2, 1), (1, -2), (1, 2), (-2, 1), (-2, -1), (-1, -2), (-1, 2)]
_KING_OFF = [(0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)]


def _in_board(c: int, r: int) -> bool:
    return 0 <= c < 8 and 0 <= r < 8


def _step_table(offsets):
    table = [0] * 64
    for sq in range(64):
        col, row = sq & 7, sq >> 3
        mask = 0
        for dc, dr in offsets:
            c, r = col + dc, row + dr
            if _in_board(c, r):
                mask |= 1 << (r * 8 + c)
        table[sq] = mask
    return table


KNIGHT_ATK = _step_table(_KNIGHT_OFF)
KING_ATK = _step_table(_KING_OFF)

# RAYS[dir][sq]: every square from sq along dir to the edge (sq excluded).
RAYS = [[0] * 64 for _ in range(8)]
for _d, (_dc, _dr) in enumerate(_DIRS):
    for _sq in range(64):
        _c, _r = (_sq & 7) + _dc, (_sq >> 3) + _dr
        _m = 0
        while _in_board(_c, _r):
            _m |= 1 << (_r * 8 + _c)
            _c += _dc
            _r += _dr
        RAYS[_d][_sq] = _m

# Pawn capture squares per colour.
PAWN_ATK = [[0] * 64, [0] * 64]
for _sq in range(64):
    _col, _row = _sq & 7, _sq >> 3
    for _dc in (-1, 1):
        if _in_board(_col + _dc, _row + 1):
            PAWN_ATK[WHITE][_sq] |= 1 << ((_row + 1) * 8 + (_col + _dc))
        if _in_board(_col + _dc, _row - 1):
            PAWN_ATK[BLACK][_sq] |= 1 << ((_row - 1) * 8 + (_col + _dc))

# Touching one of these squares (moving from it, or capturing onto it) clears the
# matching castling rights — covers king moves, rook moves, and rook captures.
_CASTLE_CLEAR = [0] * 64
_CASTLE_CLEAR[4] = WK | WQ  # e1
_CASTLE_CLEAR[7] = WK  # h1
_CASTLE_CLEAR[0] = WQ  # a1
_CASTLE_CLEAR[60] = BK | BQ  # e8
_CASTLE_CLEAR[63] = BK  # h8
_CASTLE_CLEAR[56] = BQ  # a8


def _bits(bb: int):
    """Yield the index of each set bit."""
    while bb:
        low = bb & -bb
        yield low.bit_length() - 1
        bb ^= low


def _ray_attacks(sq: int, direction: int, occ: int) -> int:
    """Squares attacked from ``sq`` along ``direction``, stopping at the first blocker."""
    attacks = RAYS[direction][sq]
    blockers = attacks & occ
    if blockers:
        if _POS[direction]:
            nearest = (blockers & -blockers).bit_length() - 1  # lowest set bit
        else:
            nearest = blockers.bit_length() - 1  # highest set bit
        attacks ^= RAYS[direction][nearest]  # drop everything beyond the blocker
    return attacks


class BitPosition:
    """Mutable bitboard position with reversible make/unmake."""

    __slots__ = ("pbb", "occ", "all", "mail", "turn", "ep", "castling")

    def __init__(self) -> None:
        self.pbb = [[0] * 6, [0] * 6]  # [colour][ptype] -> bitboard
        self.occ = [0, 0]
        self.all = 0
        self.mail = [-1] * 64  # square -> colour*6 + ptype, or -1
        self.turn = WHITE
        self.ep = -1  # en-passant target square, or -1
        self.castling = 0

    # ------------------------------------------------------------- construction
    @classmethod
    def from_game(cls, game) -> BitPosition:
        """Build from a :class:`kaisparov.core.board.ChessGame`."""
        pos = cls()
        grid = game.grid
        for col in range(8):
            column = grid[col]
            for row in range(8):
                piece = column[row]
                if piece is None:
                    continue
                color = WHITE if piece.player == Player.WHITE else BLACK
                pt = PT_IDX[piece.type]
                pos._set(color, pt, row * 8 + col)

        pos.turn = WHITE if game.turn == Player.WHITE else BLACK
        if game.en_passant_target is not None:
            ec, er = game.en_passant_target
            pos.ep = er * 8 + ec

        pos.castling = pos._castling_from_grid(grid)
        return pos

    @staticmethod
    def _castling_from_grid(grid) -> int:
        def unmoved(col, row, pt, player):
            p = grid[col][row]
            return p is not None and p.type == pt and p.player == player and not p.has_moved

        rights = 0
        if unmoved(4, 0, PieceType.KING, Player.WHITE):
            if unmoved(7, 0, PieceType.ROOK, Player.WHITE):
                rights |= WK
            if unmoved(0, 0, PieceType.ROOK, Player.WHITE):
                rights |= WQ
        if unmoved(4, 7, PieceType.KING, Player.BLACK):
            if unmoved(7, 7, PieceType.ROOK, Player.BLACK):
                rights |= BK
            if unmoved(0, 7, PieceType.ROOK, Player.BLACK):
                rights |= BQ
        return rights

    # --------------------------------------------------------------- bit helpers
    def _set(self, color: int, pt: int, sq: int) -> None:
        b = 1 << sq
        self.pbb[color][pt] |= b
        self.occ[color] |= b
        self.all |= b
        self.mail[sq] = color * 6 + pt

    def _clr(self, color: int, pt: int, sq: int) -> None:
        b = ~(1 << sq)
        self.pbb[color][pt] &= b
        self.occ[color] &= b
        self.all &= b
        self.mail[sq] = -1

    # ------------------------------------------------------------- move generation
    def gen_moves(self) -> list[tuple[int, int, int]]:
        """Every pseudo-legal ``(from_sq, to_sq, flag)`` for the side to move."""
        color = self.turn
        enemy = color ^ 1
        own = self.occ[color]
        not_own = ~own
        occ = self.all
        enemy_occ = self.occ[enemy]
        pbb = self.pbb[color]
        moves: list[tuple[int, int, int]] = []
        add = moves.append

        # Knights.
        for frm in _bits(pbb[KNIGHT_IDX]):
            for to in _bits(KNIGHT_ATK[frm] & not_own):
                add((frm, to, NORMAL))

        # King (single steps + castling).
        for frm in _bits(pbb[KING_IDX]):
            for to in _bits(KING_ATK[frm] & not_own):
                add((frm, to, NORMAL))
        self._add_castling(color, occ, add)

        # Sliders.
        for frm in _bits(pbb[BISHOP_IDX]):
            self._add_slider(frm, _BISHOP_DIRS, occ, not_own, add)
        for frm in _bits(pbb[ROOK_IDX]):
            self._add_slider(frm, _ROOK_DIRS, occ, not_own, add)
        for frm in _bits(pbb[QUEEN_IDX]):
            self._add_slider(frm, _QUEEN_DIRS, occ, not_own, add)

        # Pawns.
        self._add_pawns(color, enemy_occ, occ, add)
        return moves

    def _add_slider(self, frm, dirs, occ, not_own, add) -> None:
        targets = 0
        for d in dirs:
            targets |= _ray_attacks(frm, d, occ)
        for to in _bits(targets & not_own):
            add((frm, to, NORMAL))

    def _add_castling(self, color, occ, add) -> None:
        if color == WHITE:
            if (self.castling & WK) and not (occ & 0x60):  # f1,g1 (sq 5,6)
                add((4, 6, CASTLE_K))
            if (self.castling & WQ) and not (occ & 0x0E):  # b1,c1,d1 (sq 1,2,3)
                add((4, 2, CASTLE_Q))
        else:
            if (self.castling & BK) and not (occ & (0x60 << 56)):  # f8,g8
                add((60, 62, CASTLE_K))
            if (self.castling & BQ) and not (occ & (0x0E << 56)):  # b8,c8,d8
                add((60, 58, CASTLE_Q))

    def _add_pawns(self, color, enemy_occ, occ, add) -> None:
        pawns = self.pbb[color][PAWN_IDX]
        ep_bit = (1 << self.ep) if self.ep >= 0 else 0
        if color == WHITE:
            for frm in _bits(pawns):
                to = frm + 8
                # A pawn on the last rank (to >= 64) has no push: no promotion here.
                if to < 64 and not (occ >> to) & 1:
                    add((frm, to, NORMAL))
                    if 8 <= frm < 16 and not (occ >> (frm + 16)) & 1:
                        add((frm, frm + 16, DOUBLE))
                atk = PAWN_ATK[WHITE][frm]
                for to in _bits(atk & enemy_occ):
                    add((frm, to, NORMAL))
                if atk & ep_bit:
                    add((frm, self.ep, EPCAP))
        else:
            for frm in _bits(pawns):
                to = frm - 8
                if to >= 0 and not (occ >> to) & 1:
                    add((frm, to, NORMAL))
                    if 48 <= frm < 56 and not (occ >> (frm - 16)) & 1:
                        add((frm, frm - 16, DOUBLE))
                atk = PAWN_ATK[BLACK][frm]
                for to in _bits(atk & enemy_occ):
                    add((frm, to, NORMAL))
                if atk & ep_bit:
                    add((frm, self.ep, EPCAP))

    # --------------------------------------------------------------- make / unmake
    def make(self, move: tuple[int, int, int]) -> tuple[int, int, int, int]:
        """Apply ``move``; return an undo tuple ``(captured_code, captured_sq, ep, castling)``."""
        frm, to, flag = move
        color = self.turn
        enemy = color ^ 1
        code = self.mail[frm]
        pt = code % 6
        ep_prev = self.ep
        castle_prev = self.castling

        captured_code = -1
        captured_sq = to
        if flag == EPCAP:
            captured_sq = to - 8 if color == WHITE else to + 8
            captured_code = self.mail[captured_sq]
            self._clr(enemy, captured_code % 6, captured_sq)
        else:
            other = self.mail[to]
            if other != -1:
                captured_code = other
                self._clr(enemy, other % 6, to)

        self._clr(color, pt, frm)
        self._set(color, pt, to)

        if flag == CASTLE_K:
            rook_from, rook_to = (7, 5) if color == WHITE else (63, 61)
            self._clr(color, ROOK_IDX, rook_from)
            self._set(color, ROOK_IDX, rook_to)
        elif flag == CASTLE_Q:
            rook_from, rook_to = (0, 3) if color == WHITE else (56, 59)
            self._clr(color, ROOK_IDX, rook_from)
            self._set(color, ROOK_IDX, rook_to)

        self.ep = ((frm + to) >> 1) if flag == DOUBLE else -1
        self.castling &= ~(_CASTLE_CLEAR[frm] | _CASTLE_CLEAR[to])
        self.turn = enemy
        return (captured_code, captured_sq, ep_prev, castle_prev)

    def unmake(self, move: tuple[int, int, int], undo: tuple[int, int, int, int]) -> None:
        frm, to, flag = move
        captured_code, captured_sq, ep_prev, castle_prev = undo
        color = self.turn ^ 1  # the side that moved
        pt = self.mail[to] % 6

        self._clr(color, pt, to)
        self._set(color, pt, frm)

        if flag == CASTLE_K:
            rook_from, rook_to = (7, 5) if color == WHITE else (63, 61)
            self._clr(color, ROOK_IDX, rook_to)
            self._set(color, ROOK_IDX, rook_from)
        elif flag == CASTLE_Q:
            rook_from, rook_to = (0, 3) if color == WHITE else (56, 59)
            self._clr(color, ROOK_IDX, rook_to)
            self._set(color, ROOK_IDX, rook_from)

        if captured_code != -1:
            self._set(captured_code // 6, captured_code % 6, captured_sq)

        self.ep = ep_prev
        self.castling = castle_prev
        self.turn = color

    # ------------------------------------------------------------------ utilities
    def moves_as_coords(self) -> set[tuple[tuple[int, int], tuple[int, int]]]:
        """Move set as ``((col,row),(col,row))`` pairs, to compare with the grid engine."""
        return {((f & 7, f >> 3), (t & 7, t >> 3)) for f, t, _ in self.gen_moves()}


def perft(pos: BitPosition, depth: int) -> int:
    """Leaf count at ``depth``; a king capture ends the line (capture-the-king)."""
    if depth == 0:
        return 1
    total = 0
    for move in pos.gen_moves():
        undo = pos.make(move)
        captured_code = undo[0]
        if captured_code != -1 and captured_code % 6 == KING_IDX:
            total += 1
        else:
            total += perft(pos, depth - 1)
        pos.unmake(move, undo)
    return total


def piece_at(pos: BitPosition, sq: int) -> Piece | None:
    """Reconstruct a :class:`Piece` for square ``sq`` (mainly for debugging/interop)."""
    code = pos.mail[sq]
    if code == -1:
        return None
    color, pt = divmod(code, 6)
    player = Player.WHITE if color == WHITE else Player.BLACK
    return Piece(player, _PTYPES[pt])


__all__ = ["BitPosition", "perft", "piece_at"]
