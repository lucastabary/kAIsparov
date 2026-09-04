"""Vectorised, blocking-aware square-control maps for a *batch* of positions.

A batch of N positions is held as numpy ``uint64`` arrays (one bit per square,
``sq = row*8 + col``); every function here computes an attack/control bitboard for
all N positions at once with a handful of numpy ops, instead of one Python call
per position. Sliding attacks use the Kogge-Stone parallel-prefix fill, so there
is no per-square Python loop anywhere.

The output of :func:`attacked_by` matches :func:`kaisparov.core.rules.attacked_squares`
bit-for-bit (blocking-aware sliders that include the first blocker; pawns control
their two diagonals; knight/king control their step targets), but for N positions
at a time — the shape the batched self-play rollout needs.
"""

from __future__ import annotations

import numpy as np

U = np.uint64
FULL = U(0xFFFFFFFFFFFFFFFF)
NOT_A = U(0xFEFEFEFEFEFEFEFE)  # every square except file a
NOT_H = U(0x7F7F7F7F7F7F7F7F)  # ... except file h
NOT_AB = U(0xFCFCFCFCFCFCFCFC)  # ... except files a,b
NOT_GH = U(0x3F3F3F3F3F3F3F3F)  # ... except files g,h

_1, _2, _4, _7, _8, _9 = U(1), U(2), U(4), U(7), U(8), U(9)
_14, _16, _18, _28, _32, _36 = U(14), U(16), U(18), U(28), U(32), U(36)

WHITE, BLACK = 0, 1


# ---------------------------------------------------------------- sliding attacks
# Each fill floods the slider bits through empty squares along one direction, then
# shifts once more so the result excludes the slider's own square and includes the
# first blocker. Diagonal/lateral directions mask the wrap file after each shift.
def _nort(g, empty):
    fl = g | (empty & (g << _8))
    em = empty & (empty << _8)
    fl |= em & (fl << _16)
    em &= em << _16
    fl |= em & (fl << _32)
    return fl << _8


def _sout(g, empty):
    fl = g | (empty & (g >> _8))
    em = empty & (empty >> _8)
    fl |= em & (fl >> _16)
    em &= em >> _16
    fl |= em & (fl >> _32)
    return fl >> _8


def _east(g, empty):
    em = empty & NOT_A
    fl = g | (em & (g << _1))
    em &= em << _1
    fl |= em & (fl << _2)
    em &= em << _2
    fl |= em & (fl << _4)
    return (fl << _1) & NOT_A


def _west(g, empty):
    em = empty & NOT_H
    fl = g | (em & (g >> _1))
    em &= em >> _1
    fl |= em & (fl >> _2)
    em &= em >> _2
    fl |= em & (fl >> _4)
    return (fl >> _1) & NOT_H


def _noea(g, empty):
    em = empty & NOT_A
    fl = g | (em & (g << _9))
    em &= em << _9
    fl |= em & (fl << _18)
    em &= em << _18
    fl |= em & (fl << _36)
    return (fl << _9) & NOT_A


def _nowe(g, empty):
    em = empty & NOT_H
    fl = g | (em & (g << _7))
    em &= em << _7
    fl |= em & (fl << _14)
    em &= em << _14
    fl |= em & (fl << _28)
    return (fl << _7) & NOT_H


def _soea(g, empty):
    em = empty & NOT_A
    fl = g | (em & (g >> _7))
    em &= em >> _7
    fl |= em & (fl >> _14)
    em &= em >> _14
    fl |= em & (fl >> _28)
    return (fl >> _7) & NOT_A


def _sowe(g, empty):
    em = empty & NOT_H
    fl = g | (em & (g >> _9))
    em &= em >> _9
    fl |= em & (fl >> _18)
    em &= em >> _18
    fl |= em & (fl >> _36)
    return (fl >> _9) & NOT_H


def rook_attacks(orth, empty):
    return _nort(orth, empty) | _sout(orth, empty) | _east(orth, empty) | _west(orth, empty)


def bishop_attacks(diag, empty):
    return _noea(diag, empty) | _nowe(diag, empty) | _soea(diag, empty) | _sowe(diag, empty)


def knight_attacks(n):
    l1 = (n >> _1) & NOT_H
    l2 = (n >> _2) & NOT_GH
    r1 = (n << _1) & NOT_A
    r2 = (n << _2) & NOT_AB
    h1 = l1 | r1
    h2 = l2 | r2
    return (h1 << _16) | (h1 >> _16) | (h2 << _8) | (h2 >> _8)


def king_attacks(k):
    lateral = ((k << _1) & NOT_A) | ((k >> _1) & NOT_H)
    kk = k | lateral
    return lateral | (kk << _8) | (kk >> _8)


def pawn_attacks(p, color):
    if color == WHITE:
        return ((p << _9) & NOT_A) | ((p << _7) & NOT_H)
    return ((p >> _7) & NOT_A) | ((p >> _9) & NOT_H)


def attacked_by(orth, diag, knights, kings, pawns, occ, color):
    """Control bitboard(s) for ``color``. All inputs are ``uint64`` scalars or (N,) arrays.

    ``orth`` = rooks|queens, ``diag`` = bishops|queens; ``occ`` = all pieces.
    Returns the same shape as the inputs.
    """
    empty = FULL ^ occ
    attacks = rook_attacks(orth, empty) | bishop_attacks(diag, empty)
    attacks |= knight_attacks(knights) | king_attacks(kings) | pawn_attacks(pawns, color)
    return attacks


# ------------------------------------------------------------------- batch packing
# Per-type layer order in the packed array (see pack_positions).
ORTH, DIAG, KNIGHTS, KINGS, PAWNS, OCC = range(6)


def pack_positions(positions, color: int) -> np.ndarray:
    """Pack ``BitPosition`` objects into a (6, N) uint64 array for ``attacked_by``.

    Rows are (orth, diag, knights, kings, pawns, occ) for ``color``.
    """
    from kaisparov.core.bitboard import (
        BISHOP_IDX,
        KING_IDX,
        KNIGHT_IDX,
        PAWN_IDX,
        QUEEN_IDX,
        ROOK_IDX,
    )

    n = len(positions)
    out = np.zeros((6, n), dtype=np.uint64)
    for i, pos in enumerate(positions):
        pbb = pos.pbb[color]
        out[ORTH, i] = pbb[ROOK_IDX] | pbb[QUEEN_IDX]
        out[DIAG, i] = pbb[BISHOP_IDX] | pbb[QUEEN_IDX]
        out[KNIGHTS, i] = pbb[KNIGHT_IDX]
        out[KINGS, i] = pbb[KING_IDX]
        out[PAWNS, i] = pbb[PAWN_IDX]
        out[OCC, i] = pos.all
    return out


def attacked_by_packed(packed: np.ndarray, color: int) -> np.ndarray:
    """Control bitboards for a (6, N) packed batch; returns an (N,) uint64 array."""
    return attacked_by(
        packed[ORTH],
        packed[DIAG],
        packed[KNIGHTS],
        packed[KINGS],
        packed[PAWNS],
        packed[OCC],
        color,
    )


__all__ = [
    "attacked_by",
    "attacked_by_packed",
    "pack_positions",
    "rook_attacks",
    "bishop_attacks",
    "knight_attacks",
    "king_attacks",
    "pawn_attacks",
]
