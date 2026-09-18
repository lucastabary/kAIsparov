"""Tests for the ChessEnv."""

from __future__ import annotations

import pytest

from kaisparov.core.coords import BOARD_SIZE
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.envs.chess_env import ChessEnv


def empty_grid():
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def test_reset_and_legal_moves():
    env = ChessEnv()
    env.reset()
    assert len(env.legal_moves()) == 20  # standard opening move count
    assert not env.done


def test_capture_reward_matches_piece_value():
    grid = empty_grid()
    grid[0][0] = Piece(Player.WHITE, PieceType.ROOK)
    grid[0][5] = Piece(Player.BLACK, PieceType.QUEEN)
    grid[4][0] = Piece(Player.WHITE, PieceType.KING)
    grid[7][7] = Piece(Player.BLACK, PieceType.KING)
    grid[7][6] = Piece(Player.BLACK, PieceType.PAWN)  # so Black still has a move afterwards
    env = ChessEnv()
    env.reset(board=grid)
    result = env.step(((0, 0), (0, 5)))
    assert result.reward == pytest.approx(9.0)  # queen value
    assert not result.done


def test_promotion_is_rewarded_like_the_material_it_gains():
    grid = empty_grid()
    grid[0][6] = Piece(Player.WHITE, PieceType.PAWN)  # a7
    grid[4][0] = Piece(Player.WHITE, PieceType.KING)
    grid[6][4] = Piece(Player.BLACK, PieceType.KING)
    env = ChessEnv()
    env.reset(board=grid)
    result = env.step(((0, 6), (0, 7), PieceType.QUEEN))
    assert result.reward == pytest.approx(8.0)  # queen (9) minus the pawn it was (1)
    assert result.info["promotion"] is PieceType.QUEEN


def test_checkmate_ends_game_with_winner():
    # Back-rank mate: the black king on h8 is boxed in by its own pawns, and Ra8 is
    # check with no escape and nothing able to block or take.
    grid = empty_grid()
    grid[0][0] = Piece(Player.WHITE, PieceType.ROOK)
    grid[7][7] = Piece(Player.BLACK, PieceType.KING)
    grid[6][6] = Piece(Player.BLACK, PieceType.PAWN)
    grid[7][6] = Piece(Player.BLACK, PieceType.PAWN)
    grid[4][0] = Piece(Player.WHITE, PieceType.KING)
    env = ChessEnv()
    env.reset(board=grid)
    result = env.step(((0, 0), (0, 7)))  # Ra1-a8#
    assert result.done
    assert env.winner == Player.WHITE
    assert env.end_reason == "checkmate"


def test_illegal_move_raises():
    env = ChessEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(((0, 0), (0, 5)))  # rook blocked by own pawn


def test_step_after_done_raises():
    grid = empty_grid()
    grid[0][0] = Piece(Player.WHITE, PieceType.ROOK)
    grid[7][7] = Piece(Player.BLACK, PieceType.KING)
    grid[6][6] = Piece(Player.BLACK, PieceType.PAWN)
    grid[7][6] = Piece(Player.BLACK, PieceType.PAWN)
    grid[4][0] = Piece(Player.WHITE, PieceType.KING)
    env = ChessEnv()
    env.reset(board=grid)
    env.step(((0, 0), (0, 7)))  # mate
    with pytest.raises(RuntimeError):
        env.step(((7, 7), (7, 6)))
