"""Tests for configurable reward shaping."""

from __future__ import annotations

import chess
import pytest

from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import PieceType
from kaisparov.training.config import RewardSettings, TrainConfig
from kaisparov.training.reward import make_reward_fn


def reward_for(settings: RewardSettings, fen: str, move) -> float:
    """Play ``move`` in ``fen`` and score it, the way the rollout does.

    The reward function reads the position *after* the move plus the Undo handle, so
    it has to be given a real move rather than a synthetic capture.
    """
    game = ChessGame(board=chess.Board(fen))
    undo = game.make(*move)
    return make_reward_fn(settings)(game, undo)


def test_material_reward():
    settings = RewardSettings(material=1.0)
    # White rook on a1 takes the black queen on a6.
    taking = reward_for(settings, "7k/8/q7/8/8/8/8/R3K3 w - - 0 1", ((0, 0), (0, 5)))
    assert taking == pytest.approx(9.0)
    quiet = reward_for(settings, "7k/8/q7/8/8/8/8/R3K3 w - - 0 1", ((0, 0), (1, 0)))
    assert quiet == pytest.approx(0.0)


def test_promotion_is_worth_the_material_it_gains():
    settings = RewardSettings(material=1.0, promotion=1.0)
    queening = reward_for(settings, "8/P7/8/8/8/8/8/4K1k1 w - - 0 1", ((0, 6), (0, 7)))
    assert queening == pytest.approx(8.0)  # queen (9) minus the pawn it was (1)
    knight = reward_for(
        settings, "8/P7/8/8/8/8/8/4K1k1 w - - 0 1", ((0, 6), (0, 7), PieceType.KNIGHT)
    )
    assert knight == pytest.approx(2.0)  # underpromotion gains less


def test_promotion_can_be_switched_off():
    settings = RewardSettings(material=1.0, promotion=0.0)
    assert reward_for(settings, "8/P7/8/8/8/8/8/4K1k1 w - - 0 1", ((0, 6), (0, 7))) == 0.0


def test_step_penalty_and_checkmate():
    settings = RewardSettings(material=1.0, step_penalty=0.01, checkmate=0.5)
    quiet = reward_for(settings, "7k/8/q7/8/8/8/8/R3K3 w - - 0 1", ((0, 0), (1, 0)))
    assert quiet == pytest.approx(-0.01)

    # Back-rank mate: Ra1-a8#, the black king shut in by its own pawns.
    mate = reward_for(settings, "7k/6pp/8/8/8/8/8/R3K3 w - - 0 1", ((0, 0), (0, 7)))
    assert mate == pytest.approx(0.5 - 0.01)  # flat bonus, decoupled from material


def test_check_bonus_applied_when_opponent_in_check():
    settings = RewardSettings(material=0.0, check=0.1)
    # Ra1-a8+ checks the black king on h8 along the rank; ...Kg7 escapes, so it is a
    # check and not mate.
    checking = reward_for(settings, "7k/6p1/8/8/8/8/8/R3K3 w - - 0 1", ((0, 0), (0, 7)))
    assert checking == pytest.approx(0.1)


def test_a_mate_does_not_also_pay_the_check_bonus():
    settings = RewardSettings(material=0.0, check=0.1, checkmate=1.0)
    mate = reward_for(settings, "7k/6pp/8/8/8/8/8/R3K3 w - - 0 1", ((0, 0), (0, 7)))
    assert mate == pytest.approx(1.0)


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
