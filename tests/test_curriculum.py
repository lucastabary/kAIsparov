"""Tests for the piece-count curriculum, focused on the king-safety guarantee."""

from __future__ import annotations

from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player
from kaisparov.core.rules import find_king, is_in_check
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum


def _count(grid, player) -> int:
    return sum(
        1
        for col in range(BOARD_SIZE)
        for row in range(BOARD_SIZE)
        if grid[col][row] is not None and grid[col][row].player == player
    )


def test_ensure_kings_safe_never_starts_in_check():
    # A dense 16-piece phase is exactly where random placement used to hang a king.
    phase = PhaseConfig(name="dense", max_pieces_per_side=16, ensure_kings_safe=True)
    curriculum = PieceCountCurriculum(phase, seed=0)
    for _ in range(200):
        grid = curriculum.get_initial_board()
        assert find_king(grid, Player.WHITE) is not None
        assert find_king(grid, Player.BLACK) is not None
        # The whole point: no side can grab a king on move 1.
        assert not is_in_check(grid, Player.WHITE)
        assert not is_in_check(grid, Player.BLACK)


def test_pieces_stay_in_their_half_and_within_budget():
    phase = PhaseConfig(name="dense", max_pieces_per_side=16, ensure_kings_safe=True)
    curriculum = PieceCountCurriculum(phase, seed=1)
    for _ in range(50):
        grid = curriculum.get_initial_board()
        assert _count(grid, Player.WHITE) <= 16
        assert _count(grid, Player.BLACK) <= 16
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                piece = grid[col][row]
                if piece is None:
                    continue
                if piece.type == PieceType.PAWN:
                    # Pawns use the shared pawn rows (never a back rank) and may sit in
                    # either half — a pre-existing quirk of the piece-count curriculum.
                    assert 0 < row < BOARD_SIZE - 1
                elif piece.player == Player.WHITE:
                    assert row < BOARD_SIZE // 2  # non-pawn white in the bottom half
                else:
                    assert row >= BOARD_SIZE // 2  # non-pawn black in the top half


def test_flag_off_preserves_raw_random_placement():
    # With the guarantee disabled the board is still valid (both kings present); we
    # don't assert a check appears (it's random), only that the flag path runs.
    phase = PhaseConfig(name="raw", max_pieces_per_side=16, ensure_kings_safe=False)
    curriculum = PieceCountCurriculum(phase, seed=2)
    grid = curriculum.get_initial_board()
    assert find_king(grid, Player.WHITE) is not None
    assert find_king(grid, Player.BLACK) is not None
