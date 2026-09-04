"""The vectorised control map must equal rules.attacked_squares, per position.

core.rules.attacked_squares is the reference; core.bitboard_batch.attacked_by must
reproduce it bit-for-bit, for a whole batch at once, for both colours.
"""

from __future__ import annotations

import random

from kaisparov.core import bitboard_batch as bbb
from kaisparov.core.bitboard import BitPosition
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves
from kaisparov.core.pieces import PieceType, Player
from kaisparov.core.rules import attacked_squares


def _coords_to_bb(coords) -> int:
    bb = 0
    for cx, cy in coords:
        bb |= 1 << (cy * 8 + cx)
    return bb


def _collect(n_target: int, seed: int = 7):
    rng = random.Random(seed)
    games, positions = [], []
    while len(positions) < n_target:
        game = ChessGame()
        for _ in range(60):
            moves = all_moves(game.grid, game.turn, game.en_passant_target)
            games.append(game.copy())
            positions.append(BitPosition.from_game(game))
            if len(positions) >= n_target or not moves:
                break
            src, dest = rng.choice(moves)
            undo = game.make(src, dest)
            if undo.captured is not None and undo.captured.type == PieceType.KING:
                break
    return games[:n_target], positions[:n_target]


def test_vectorised_control_map_matches_reference():
    games, positions = _collect(400)
    for color, player in ((bbb.WHITE, Player.WHITE), (bbb.BLACK, Player.BLACK)):
        packed = bbb.pack_positions(positions, color)
        got = bbb.attacked_by_packed(packed, color)
        for i, game in enumerate(games):
            assert int(got[i]) == _coords_to_bb(attacked_squares(game.grid, player))


def test_attacked_by_accepts_scalar_inputs():
    # The same functions must work on a single position (0-d uint64), not only arrays.
    import numpy as np

    game = ChessGame()
    pos = BitPosition.from_game(game)
    packed = bbb.pack_positions([pos], bbb.WHITE)
    single = bbb.attacked_by(
        np.uint64(packed[bbb.ORTH, 0]),
        np.uint64(packed[bbb.DIAG, 0]),
        np.uint64(packed[bbb.KNIGHTS, 0]),
        np.uint64(packed[bbb.KINGS, 0]),
        np.uint64(packed[bbb.PAWNS, 0]),
        np.uint64(packed[bbb.OCC, 0]),
        bbb.WHITE,
    )
    assert int(single) == _coords_to_bb(attacked_squares(game.grid, Player.WHITE))
