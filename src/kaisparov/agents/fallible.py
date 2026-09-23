"""Make any agent fallible: now and then, a uniformly random legal move.

``Fallible(agent, random_move_prob)`` plays ``agent``'s move, except that on each move,
with probability ``random_move_prob``, it plays a random legal move instead. A strong
opponent that sometimes blunders is a graded one: the learner can beat it by punishing
the blunders long before it could beat the opponent itself, and the probability is
the knob that sets how often. Pool entries take it as ``random_move_prob`` (default 0,
see config/pools.yaml).
"""

from __future__ import annotations

import random

from kaisparov.agents.base import Move, Policy
from kaisparov.core.game import ChessGame


class Fallible:
    def __init__(self, agent: Policy, random_move_prob: float, seed: int | None = None):
        if not 0.0 <= random_move_prob <= 1.0:
            raise ValueError(f"random_move_prob must be in [0, 1], got {random_move_prob}")
        self.agent = agent
        self.random_move_prob = random_move_prob
        self.name = f"{getattr(agent, 'name', 'agent')}+random{random_move_prob:g}"
        self._rng = random.Random(seed)

    def select_move(self, game: ChessGame) -> Move | None:
        if self._rng.random() < self.random_move_prob:
            moves = game.legal_moves()
            return self._rng.choice(moves) if moves else None
        return self.agent.select_move(game)


__all__ = ["Fallible"]
