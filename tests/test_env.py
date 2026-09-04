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
    grid[7][6] = Piece(Player.BLACK, PieceType.PAWN)  # so Black still has a move afterwards
    env = ChessEnv()
    env.reset(board=grid)
    result = env.step(((0, 0), (0, 5)))
    assert result.reward == pytest.approx(9.0)  # queen value
    assert not result.done


def test_king_capture_ends_game_with_winner():
    grid = empty_grid()
    grid[0][0] = Piece(Player.WHITE, PieceType.ROOK)
    grid[3][0] = Piece(Player.BLACK, PieceType.KING)
    env = ChessEnv()
    env.reset(board=grid)
    result = env.step(((0, 0), (3, 0)))
    assert result.done
    assert env.winner == Player.WHITE
    assert result.info["captured"].type == PieceType.KING


def test_illegal_move_raises():
    env = ChessEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(((0, 0), (0, 5)))  # rook blocked by own pawn


def test_step_after_done_raises():
    grid = empty_grid()
    grid[0][0] = Piece(Player.WHITE, PieceType.ROOK)
    grid[3][0] = Piece(Player.BLACK, PieceType.KING)
    env = ChessEnv()
    env.reset(board=grid)
    env.step(((0, 0), (3, 0)))
    with pytest.raises(RuntimeError):
        env.step(((3, 0), (3, 1)))
