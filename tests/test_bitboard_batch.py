"""The vectorised control map must equal rules.attacked_squares, per position.

core.rules.attacked_squares is the reference; core.bitboard_batch.attacked_by must
reproduce it bit-for-bit, for a whole batch at once, for both colours.
"""

from __future__ import annotations

import random

from kaisparov.core import bitboard_batch as bbb
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import Player
from kaisparov.core.rules import attacked_squares


def _coords_to_bb(coords) -> int:
    bb = 0
    for cx, cy in coords:
        bb |= 1 << (cy * 8 + cx)
    return bb


def _collect(n_target: int, seed: int = 7):
    """Random self-play positions, sampled along whole games rather than one by one."""
    rng = random.Random(seed)
    games = []
    while len(games) < n_target:
        game = ChessGame()
        for _ in range(60):
            games.append(game.copy())
            moves = game.legal_moves()
            if len(games) >= n_target or not moves:
                break
            game.make(*rng.choice(moves))
    return games[:n_target]


def test_vectorised_control_map_matches_reference():
    games = _collect(400)
    for color, player in ((bbb.WHITE, Player.WHITE), (bbb.BLACK, Player.BLACK)):
        packed = bbb.pack_boards([g.board for g in games], color)
        got = bbb.attacked_by_packed(packed, color)
        for i, game in enumerate(games):
            assert int(got[i]) == _coords_to_bb(attacked_squares(game, player))


def test_attacked_by_accepts_scalar_inputs():
    # The same functions must work on a single position (0-d uint64), not only arrays.
    import numpy as np

    game = ChessGame()
    packed = bbb.pack_boards([game.board], bbb.WHITE)
    single = bbb.attacked_by(
        np.uint64(packed[bbb.ORTH, 0]),
        np.uint64(packed[bbb.DIAG, 0]),
        np.uint64(packed[bbb.KNIGHTS, 0]),
        np.uint64(packed[bbb.KINGS, 0]),
        np.uint64(packed[bbb.PAWNS, 0]),
        np.uint64(packed[bbb.OCC, 0]),
        bbb.WHITE,
    )
    assert int(single) == _coords_to_bb(attacked_squares(game, Player.WHITE))
