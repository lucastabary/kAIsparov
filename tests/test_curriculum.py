"""Tests for the piece-count curriculum, focused on the king-safety guarantee."""

from __future__ import annotations

import pytest

from kaisparov.core.game import ChessGame
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


def test_curriculum_never_leaves_the_side_not_to_move_in_check():
    """White moves first, so Black in check is a position chess cannot reach.

    python-chess would generate the capture of the black king there, which the
    material reward scores at the king's sentinel value — a silent exploit. The
    sampler must redraw instead, even with ensure_kings_safe off (which on its own
    leaves about a third of the draws illegal).
    """
    for safe in (True, False):
        phase = PhaseConfig(name="t", max_pieces_per_side=6, ensure_kings_safe=safe)
        curriculum = PieceCountCurriculum(phase, seed=0)
        for _ in range(200):
            game = ChessGame(initial_board=curriculum.get_initial_board())
            assert not game.is_in_check(Player.BLACK)


# ------------------------------------------------------------ lopsided (won) endgames


def test_defender_pieces_draws_a_won_endgame_for_a_random_strong_side():
    phase = PhaseConfig(
        name="won",
        max_pieces_per_side=3,
        allow_minor=False,
        allow_pawns=False,
        defender_pieces=1,
    )
    curriculum = PieceCountCurriculum(phase, seed=0)
    sides = set()
    for _ in range(100):
        grid, strong = curriculum.get_start()
        assert strong is not None
        sides.add(strong)
        weak = strong.opponent
        assert _count(grid, strong) == 3  # king + two majors
        assert _count(grid, weak) == 1  # the bare king
        assert all(
            grid[col][row].type in (PieceType.KING, PieceType.ROOK, PieceType.QUEEN)
            for col in range(BOARD_SIZE)
            for row in range(BOARD_SIZE)
            if grid[col][row] is not None
        )
        assert not ChessGame(initial_board=grid).is_in_check(Player.BLACK)
    assert sides == {Player.WHITE, Player.BLACK}  # the learner trains as both colours


def test_a_balanced_phase_leaves_the_side_to_the_rollout():
    curriculum = PieceCountCurriculum(PhaseConfig(name="t", max_pieces_per_side=4), seed=0)
    _, strong = curriculum.get_start()
    assert strong is None


def test_defender_pieces_out_of_range_is_refused():
    for bad in (0, 5):
        with pytest.raises(ValueError, match="defender_pieces"):
            PhaseConfig(name="t", max_pieces_per_side=4, defender_pieces=bad)
