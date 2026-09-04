"""The bitboard move generator must match the grid engine exactly.

The grid engine (core.board + core.movegen) is the reference. These tests pin the
bitboard engine to it via perft node counts and a move-set equivalence check over
real played positions (which exercises castling, en passant and capture-the-king).
"""

from __future__ import annotations

import random

from kaisparov.core import bitboard as bb
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves
from kaisparov.core.pieces import PieceType


def _grid_perft(game: ChessGame, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for src, dest in all_moves(game.grid, game.turn, game.en_passant_target):
        undo = game.make(src, dest)
        if undo.captured is not None and undo.captured.type == PieceType.KING:
            total += 1
        else:
            total += _grid_perft(game, depth - 1)
        game.unmake(undo)
    return total


def test_bitboard_perft_matches_known_oracle():
    # The capture-the-king perft(4) from the standard start; the whole engine's
    # correctness anchor (see also the grid engine).
    assert bb.perft(bb.BitPosition.from_game(ChessGame()), 4) == 197742


def test_bitboard_perft_matches_grid_shallow():
    for depth in (1, 2, 3):
        assert bb.perft(bb.BitPosition.from_game(ChessGame()), depth) == _grid_perft(
            ChessGame(), depth
        )


def test_bitboard_moves_match_grid_over_random_games():
    rng = random.Random(1234)
    checked = 0
    for _ in range(40):
        game = ChessGame()
        for _ in range(60):
            grid_set = set(all_moves(game.grid, game.turn, game.en_passant_target))
            assert bb.BitPosition.from_game(game).moves_as_coords() == grid_set
            checked += 1
            if not grid_set:
                break
            src, dest = rng.choice(list(grid_set))
            undo = game.make(src, dest)
            if undo.captured is not None and undo.captured.type == PieceType.KING:
                break
    assert checked > 500  # sanity: the walk actually explored many positions
