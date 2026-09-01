"""Tests for configurable reward shaping."""

from __future__ import annotations

import pytest

from kaisparov.core.board import ChessGame
from kaisparov.core.coords import BOARD_SIZE
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.training.config import RewardSettings, TrainConfig
from kaisparov.training.reward import make_reward_fn


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def test_material_reward():
    rf = make_reward_fn(RewardSettings(material=1.0))
    assert rf(empty_game(), Piece(Player.BLACK, PieceType.QUEEN)) == pytest.approx(0.09)
    assert rf(empty_game(), None) == pytest.approx(0.0)


def test_step_penalty_and_king_capture():
    rf = make_reward_fn(RewardSettings(material=1.0, step_penalty=0.01, king_capture=0.5))
    assert rf(empty_game(), None) == pytest.approx(-0.01)
    # king value 1.0 * material + king_capture - step_penalty
    got = rf(empty_game(), Piece(Player.BLACK, PieceType.KING))
    assert got == pytest.approx(1.0 + 0.5 - 0.01)


def test_check_bonus_applied_when_opponent_in_check():
    game = empty_game(turn=Player.WHITE)
    game.grid[4][0] = Piece(Player.WHITE, PieceType.KING)
    game.grid[4][5] = Piece(Player.BLACK, PieceType.ROOK)  # attacks the white king down the file
    assert game.is_in_check(Player.WHITE)

    rf = make_reward_fn(RewardSettings(material=0.0, check=0.1))
    assert rf(game, None) == pytest.approx(0.1)


# ----------------------------------------------------------------- config plumbing
def test_reward_inline_config():
    c = TrainConfig.from_dict({"reward": {"material": 0.5, "step_penalty": 0.01}})
    assert c.reward.material == 0.5
    assert c.reward.step_penalty == 0.01
    assert c.reward.check == 0.0  # untouched default


def test_reward_preset_from_file():
    c = TrainConfig.from_dict({"reward": "aggressive"})
    assert c.reward.preset == "aggressive"
    assert c.reward.check > 0.0


def test_unknown_reward_preset_raises():
    with pytest.raises(ValueError):
        TrainConfig.from_dict({"reward": "does_not_exist"})
