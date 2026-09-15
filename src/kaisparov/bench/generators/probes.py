"""Probes of the model's heads: does the value know who wins, does the policy rank
the right move high even when it does not play it?

- ``value_sign`` — positions the oracle can call won (a forced king capture within two
  moves) or lost (every move lets the opponent take the king), half and half;
- ``policy_rank`` — any move-finding theme, re-graded on where the policy ranks the
  answer. It wraps another generator instead of sampling on its own, so the problems
  are exactly that theme's.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from kaisparov.bench.generators.base import (
    HEAVY_TYPES,
    PIECE_TYPES,
    BoardBuilder,
    GenerationError,
    ProblemGenerator,
    SamplingGenerator,
)
from kaisparov.bench.oracle import Oracle
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import FindMove, PolicyRank, ValueSign, WinMaterial
from kaisparov.core.pieces import Player


class ValueSignGenerator(SamplingGenerator):
    name = "value_sign"
    theme = "value_sign"
    description = "Won or lost by force (oracle): the value estimate must have the right sign."

    def __init__(self, depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        want = rng.choice((1, -1))
        # The side to move gets the heavy pieces when it should be winning, the enemy
        # gets them when it should be losing — otherwise the rejection rate explodes.
        strong = [rng.choice(HEAVY_TYPES) for _ in range(rng.randint(2, 4))]
        weak = [rng.choice(PIECE_TYPES) for _ in range(rng.randint(0, 3))]
        white, black = (strong, weak) if want > 0 else (weak, strong)
        board = BoardBuilder.random_board(rng, white, black)
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if self.oracle.is_over(game):
            return None
        if want > 0:
            # Aim at each depth equally often: left alone, "the king is en prise" (depth
            # 1) would crowd out the wins the value head has to look ahead for.
            depth = rng.randint(1, self.depth)
            if self.oracle.win_depth(game, self.depth) != depth:
                return None
        else:
            if game.is_in_check(Player.BLACK) or not self.oracle.loses_within(game, 1):
                return None
            depth = 1
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=ValueSign(expected=want),
            difficulty=depth,
            tags=("won" if want > 0 else "lost",),
        )


class PolicyRankGenerator(ProblemGenerator):
    """Another generator's problems, asked of the policy's ranking instead of its move."""

    name = "policy_rank"
    theme = "policy_rank"
    description = "Where does the policy rank the answer of another theme (top-k)?"

    def __init__(self, source: str, source_params: dict[str, Any] | None = None, k: int = 3):
        if source == self.name:
            raise ValueError("policy_rank cannot wrap itself")
        self.source = ProblemGenerator.create(source, source_params)
        self.source_name = source
        self.k = k

    def generate(self, count: int | None, rng: random.Random) -> list[Problem]:
        problems = []
        for problem in self.source.generate(count, rng):
            task = problem.task
            if isinstance(task, FindMove):
                accepted = task.accepted
            elif isinstance(task, WinMaterial) and task.reference:
                accepted = task.reference
            else:
                raise GenerationError(
                    f"policy_rank: {self.source_name} problems ({task.kind}) have no answer to rank"
                )
            problems.append(
                replace(
                    problem,
                    theme=f"policy_{problem.theme}",
                    task=PolicyRank(accepted=accepted, k=self.k),
                    generator=self.name,
                    tags=(*problem.tags, self.source_name),
                )
            )
        return problems


__all__ = ["PolicyRankGenerator", "ValueSignGenerator"]
