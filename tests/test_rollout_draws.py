"""Training rollouts end a stalemate as a draw — never as the win it used to be.

Before the stalemate rule, a stalemated side had to play a king-hanging move, so
stalemating the opponent was worth a king capture one ply later. These tests script
the learner's move so the position is deterministic, whatever the untrained network
would have picked.
"""

from __future__ import annotations

import dataclasses
import random
import types

import torch

from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player
from kaisparov.models import rgcn
from kaisparov.models.factory import load_backend_spec
from kaisparov.training.config import RewardSettings
from kaisparov.training.curriculum import BaseCurriculum
from kaisparov.training.reward import make_reward_fn
from kaisparov.training.rollout import collect_data
from kaisparov.training.rollout_vs import collect_vs_opponent

W, B = Player.WHITE, Player.BLACK

# White's queen to c7 boxes in the lone black king on a8 without attacking it.
STALEMATING_MOVE = ((2, 0), (2, 6))

# A king capture that would be worth a lot if the stalemate were mistaken for a win.
REWARDS = RewardSettings(material=1.0, checkmate=10.0, step_penalty=0.0)


class StalemateSetup(BaseCurriculum):
    """Always the same board: White to move, one queen move from stalemating Black."""

    def get_initial_board(self):
        grid: list[list[Piece | None]] = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        grid[7][0] = Piece(W, PieceType.KING)
        grid[2][0] = Piece(W, PieceType.QUEEN)
        grid[0][7] = Piece(B, PieceType.KING)
        return grid


class ScriptedProcessor(rgcn.RGCNProcessor):
    """The real processor, except the chosen move is always the stalemating one."""

    def process_output(self, model_output, game, deterministic, legal_mask=None):
        action = super().process_output(model_output, game, deterministic, legal_mask)
        return dataclasses.replace(action, move_coords=STALEMATING_MOVE)


SCRIPTED = types.SimpleNamespace(
    PROCESSOR_CLASS=ScriptedProcessor, get_legal_mask=rgcn.get_legal_mask
)


def _agent_and_buffer(self_play: bool):
    spec = load_backend_spec("rgcn")
    agent = spec.model_class.create_agent(device=torch.device("cpu"), hidden_dim=8)
    buffer = spec.buffer_class(gamma=0.99, gae_lambda=0.95, self_play=self_play)
    return agent, buffer


class NeverAsked:
    """An opponent the stalemated side must not get to move for."""

    name = "never-asked"

    def select_move(self, game):
        raise AssertionError("the opponent was asked to move out of a stalemate")


def test_self_play_ends_a_stalemate_as_a_draw_worth_nothing():
    agent, buffer = _agent_and_buffer(self_play=True)
    stats = collect_data(
        agent,
        ChessGame(),
        buffer,
        num_episodes=1,
        max_steps_per_episode=20,
        model_module=SCRIPTED,
        curriculum=StalemateSetup(),
        reward_fn=make_reward_fn(REWARDS),
    )

    assert stats["stalemate_rate"] == 1.0
    assert stats["checkmate_rate"] == 0.0
    assert buffer.rewards == [0.0]
    assert buffer.dones == [True]


def test_league_ends_the_learners_stalemate_before_the_opponent_replies():
    # The learner's colour is drawn from the seed: pick one that seats it as White,
    # the side holding the stalemating move.
    seed = next(s for s in range(100) if random.Random(s).choice([W, B]) == W)
    agent, buffer = _agent_and_buffer(self_play=False)
    stats = collect_vs_opponent(
        agent,
        buffer,
        num_episodes=1,
        max_steps_per_episode=20,
        model_module=SCRIPTED,
        curriculum=StalemateSetup(),
        reward_settings=REWARDS,
        opponent=NeverAsked(),
        seed=seed,
    )

    assert stats["drawrate"] == 1.0
    assert stats["winrate"] == 0.0
    assert buffer.rewards == [0.0]  # no king-capture bonus for a draw
    assert buffer.dones == [True]
