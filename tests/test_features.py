"""The node feature sets are frozen: a name used by a run never changes meaning.

A checkpoint records the *name* of the set it was trained on (``features:`` in its
run's config), not the encoding itself. If ``pieces`` started meaning something else,
every model trained on it would still load — and then read garbage. So each set's
output is pinned here, on fixed positions, one 64-bit square mask per feature column
(bit ``row * 8 + col``, a1 = bit 0).

If one of these fails, you changed what an existing set means. Don't update the
expected values: add a new set under a new name in ``models/features.py``.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from kaisparov.core.game import ChessGame
from kaisparov.models.features import DEFAULT_FEATURES, FEATURE_SETS

FENS = {
    "start": chess.STARTING_FEN,
    # Black to move: "own" is now Black, and the columns swap sides.
    "black_to_move": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # The bishop on b4 checks the king on e1 through c3 and d2...
    "check": "4k3/8/8/8/1b6/8/2P5/4K2R w K - 0 1",
    # ...unless a pawn stands on d2: the line stops there, and e1 is safe.
    "blocked": "4k3/8/8/8/1b6/8/3P4/4K2R w K - 0 1",
}

# Columns 0-5: the mover's king, queen, bishops, rooks, knights, pawns; 6-11: the
# opponent's, same order.
PIECES = {
    "start": [
        0x10, 0x8, 0x24, 0x81, 0x42, 0xFF00,
        0x1000000000000000, 0x800000000000000, 0x2400000000000000,
        0x8100000000000000, 0x4200000000000000, 0xFF000000000000,
    ],
    "black_to_move": [
        0x1000000000000000, 0x800000000000000, 0x2400000000000000,
        0x8100000000000000, 0x4200000000000000, 0xFF000000000000,
        0x10, 0x8, 0x24, 0x81, 0x42, 0x1000EF00,
    ],
    "check": [0x10, 0x0, 0x0, 0x80, 0x0, 0x400, 0x1000000000000000, 0x0, 0x2000000, 0x0, 0x0, 0x0],
    "blocked": [0x10, 0x0, 0x0, 0x80, 0x0, 0x800, 0x1000000000000000, 0x0, 0x2000000, 0x0, 0x0, 0x0],
}  # fmt: skip

# pieces_control adds column 12, attacked by the opponent, and column 13, controlled
# by the mover.
CONTROL = {
    "start": [0x7EFFFF0000000000, 0xFFFF7E],
    "black_to_move": [0x1AA44FFFF7E, 0x7EFFFF0000000000],
    "check": [0x2838080500050810, 0x80808080808AB878],
    "blocked": [0x2838080500050800, 0x808080808094B878],
}

EXPECTED = {
    "pieces": {name: PIECES[name] for name in FENS},
    "pieces_control": {name: PIECES[name] + CONTROL[name] for name in FENS},
}


def _masks(x: np.ndarray) -> list[int]:
    """``(64, dim)`` features -> one square bitmask per column."""
    return [sum(1 << sq for sq in range(64) if x[sq, col]) for col in range(x.shape[1])]


def test_every_registered_set_is_pinned_here():
    assert set(EXPECTED) == set(FEATURE_SETS), "a new set needs its expected output here"


@pytest.mark.parametrize("name", sorted(EXPECTED))
@pytest.mark.parametrize("position", sorted(FENS))
def test_a_feature_set_encodes_exactly_what_it_always_has(name, position):
    feature_set = FEATURE_SETS[name]
    game = ChessGame(board=chess.Board(FENS[position]))
    single = feature_set.encode(game)
    batched = feature_set.encode_batch([game])[0]

    assert single.shape == (64, feature_set.dim)
    assert set(np.unique(single)) <= {0.0, 1.0}
    assert _masks(single) == EXPECTED[name][position]
    assert np.array_equal(batched, single)


def test_the_control_flags_see_a_blocked_line():
    e1 = 1 << 4
    assert CONTROL["check"][0] & e1  # the king's square is attacked: it is in check
    assert not CONTROL["blocked"][0] & e1  # the pawn on d2 cuts the diagonal


def test_new_runs_train_on_piece_one_hots_only():
    assert DEFAULT_FEATURES == "pieces"
