"""Numeric checks for the negamax self-play GAE."""

from __future__ import annotations

from kaisparov.training.ppo import PPOBuffer


def _fill(buf: PPOBuffer, values, rewards, dones):
    buf.values = list(values)
    buf.rewards = list(rewards)
    buf.dones = list(dones)
    buf.actions = [0] * len(values)  # __len__ reads actions


def test_negamax_penalizes_the_losing_move():
    # step0: White (no capture). step1: Black captures White's king (+1, terminal).
    buf = PPOBuffer(gamma=1.0, gae_lambda=1.0, self_play=True)
    _fill(buf, values=[0.0, 0.0], rewards=[0.0, 1.0], dones=[False, True])
    buf.compute_returns_and_advantages()
    assert buf.advantages[1] == 1.0  # the winning move is good for its player
    assert buf.advantages[0] < 0  # the move that allowed it is penalized


def test_single_perspective_does_not_flip():
    buf = PPOBuffer(gamma=1.0, gae_lambda=1.0, self_play=False)
    _fill(buf, values=[0.0, 0.0], rewards=[0.0, 1.0], dones=[False, True])
    buf.compute_returns_and_advantages()
    assert buf.advantages[0] > 0  # reward propagates without perspective flip
