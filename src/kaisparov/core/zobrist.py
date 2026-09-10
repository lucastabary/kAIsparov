"""Zobrist hashing: a 64-bit fingerprint of a position, updatable in O(1).

Two positions share a fingerprint iff they have the same pieces on the same
squares, the same side to move, the same en-passant target *and* the same
castling rights — which is exactly the notion of "same position" the repetition
rule in :mod:`kaisparov.core.draw` needs.

Castling rights are folded in through the piece keys: there is one key per
``(player, type, square, has_moved)``, so a rook that has already moved is a
different piece from one that has not. That keeps the update in
:meth:`kaisparov.core.board.ChessGame.make` to a handful of XORs and needs no
separate rights bookkeeping. It is deliberately conservative: two positions that
differ only in a ``has_moved`` flag nobody can still use (a rook long gone from
its home square) hash differently, which can only ever *delay* a repetition
claim, never invent one.

The keys are drawn from a fixed seed, so a given position hashes to the same
value in every process — workers and the parent agree without sharing state.
"""

from __future__ import annotations

import random

from kaisparov.core.coords import ALL_SQUARES, Coord, coord_to_index
from kaisparov.core.pieces import BOARD_SIZE, NUM_PIECE_CODES, Piece, Player

Grid = list[list["Piece | None"]]

_SQUARES = BOARD_SIZE * BOARD_SIZE
_rng = random.Random(0x6B41_5350)  # fixed seed: reproducible fingerprints


def _keys(n: int) -> list[int]:
    return [_rng.getrandbits(64) for _ in range(n)]


# PIECE_KEYS[piece.code] -> (keys per square when unmoved, keys per square once moved).
# Indexed by the dense piece code rather than by (player, type): the tables are read on
# every ``make``, and hashing two enum members there costs more than the XOR it feeds.
PIECE_KEYS: list[tuple[list[int], list[int]]] = [
    (_keys(_SQUARES), _keys(_SQUARES)) for _ in range(NUM_PIECE_CODES)
]

# XORed in whenever the side to move flips.
TURN_KEY: int = _rng.getrandbits(64)

# One key per possible en-passant target square; index 0 means "no target".
EN_PASSANT_KEYS: list[int] = [0, *_keys(_SQUARES)]


def piece_key(piece: Piece, square: Coord, has_moved: bool | None = None) -> int:
    """Key for ``piece`` standing on ``square``.

    ``has_moved`` overrides the piece's own flag, which callers need when
    hashing a piece *out* of the square it occupied before the flag was set.
    """
    moved = piece.has_moved if has_moved is None else has_moved
    return PIECE_KEYS[piece.code][moved][coord_to_index(square)]


def en_passant_key(target: Coord | None) -> int:
    return 0 if target is None else EN_PASSANT_KEYS[coord_to_index(target) + 1]


def hash_position(grid: Grid, turn: Player, en_passant_target: Coord | None) -> int:
    """Hash a whole position from scratch (used once per game, then updated)."""
    key = en_passant_key(en_passant_target)
    if turn == Player.BLACK:
        key ^= TURN_KEY
    for square in ALL_SQUARES:
        piece = grid[square[0]][square[1]]
        if piece is not None:
            key ^= piece_key(piece, square)
    return key


__all__ = ["hash_position", "piece_key", "en_passant_key", "TURN_KEY"]
