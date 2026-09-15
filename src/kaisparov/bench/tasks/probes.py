"""Probe tasks: read what the contestant *thinks* instead of what it plays.

A move is the end of a pipeline — a policy distribution, maybe a search on top. Probes
look one step earlier, through the contestant's
:class:`~kaisparov.insights.Analyzer` (``TaskContext.analyzer``): does the value head
know who is winning, and how much probability does the policy put on the right move
even when it is not the argmax? That separates "the model cannot see it" from "the
model sees it and still plays something else", which a move alone cannot.

A contestant without an analyzer (the random baseline) skips probes rather than
failing them.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from kaisparov.agents.base import Policy
from kaisparov.bench.position import mirror_move, move_to_uci, uci_to_move
from kaisparov.bench.tasks.base import Outcome, Task, TaskContext, legal_uci
from kaisparov.core.board import ChessGame
from kaisparov.insights import PositionAnalysis


class ProbeTask(Task):
    """A task graded on the contestant's analysis of the position."""

    def attempt(self, game: ChessGame, policy: Policy, context: TaskContext) -> Outcome:
        if context.analyzer is None:
            return Outcome.skip("no analyzer")
        start = time.perf_counter()
        try:
            analysis = context.analyzer.analyze(game)
        except Exception as exc:  # a contestant bug must not abort the benchmark
            return Outcome.failure(f"{type(exc).__name__}: {exc}")
        seconds = time.perf_counter() - start
        if analysis is None:
            return Outcome.failure("no analysis", seconds=seconds)
        return self.read(game, analysis, seconds)

    @abstractmethod
    def read(self, game: ChessGame, analysis: PositionAnalysis, seconds: float) -> Outcome: ...


@dataclass(frozen=True)
class ValueSign(ProbeTask):
    """The value estimate must have the sign of the truth: ``+1`` won, ``-1`` lost."""

    kind: ClassVar[str] = "value_sign"

    expected: int

    def __post_init__(self) -> None:
        if self.expected not in (1, -1):
            raise ValueError("value_sign expects +1 (side to move wins) or -1 (it loses)")

    def read(self, game: ChessGame, analysis: PositionAnalysis, seconds: float) -> Outcome:
        if analysis.value is None:
            return Outcome.skip("no value estimate")
        solved = analysis.value * self.expected > 0
        return Outcome(
            score=float(solved),
            solved=solved,
            seconds=seconds,
            details={"value": round(analysis.value, 4)},
        )

    # The value is from the side to move's point of view, which the mirror keeps.
    def params(self) -> dict[str, Any]:
        return {"expected": self.expected}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> ValueSign:
        return cls(expected=int(params["expected"]))

    def describe(self) -> str:
        return "value says " + ("winning" if self.expected > 0 else "losing")


@dataclass(frozen=True)
class PolicyRank(ProbeTask):
    """An ``accepted`` move must rank in the top ``k`` of the contestant's candidates.

    The score is the probability mass the ranking puts on the accepted moves, read off
    the candidates' relative weights — so it assumes the analyzer ranks every legal
    move, which the bench's analyzers do. A move missing from the ranking counts as
    unranked.
    """

    kind: ClassVar[str] = "policy_rank"

    accepted: tuple[str, ...]
    k: int = 3

    def __post_init__(self) -> None:
        if not self.accepted or self.k < 1:
            raise ValueError("policy_rank needs accepted moves and k >= 1")
        object.__setattr__(self, "accepted", tuple(sorted(set(self.accepted))))

    def read(self, game: ChessGame, analysis: PositionAnalysis, seconds: float) -> Outcome:
        ranked = [move_to_uci(c.move) for c in analysis.candidates]
        total = sum(max(0.0, c.score) for c in analysis.candidates) or 1.0
        mass = sum(
            max(0.0, c.score) for c in analysis.candidates if move_to_uci(c.move) in self.accepted
        )
        rank = next((i for i, move in enumerate(ranked, 1) if move in self.accepted), None)
        solved = rank is not None and rank <= self.k
        return Outcome(
            score=mass / total,
            solved=solved,
            move=ranked[0] if ranked else None,
            seconds=seconds,
            details={"rank": rank, "mass": round(mass / total, 4)},
        )

    def validate(self, game: ChessGame) -> None:
        legal = legal_uci(game)
        for move in self.accepted:
            if move not in legal:
                raise ValueError(f"{self.kind}: {move} is not a legal move here")

    def mirrored(self) -> PolicyRank:
        return PolicyRank(
            tuple(move_to_uci(mirror_move(uci_to_move(m))) for m in self.accepted), self.k
        )

    def params(self) -> dict[str, Any]:
        return {"accepted": list(self.accepted), "k": self.k}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> PolicyRank:
        return cls(accepted=tuple(params["accepted"]), k=int(params.get("k", 3)))

    def describe(self) -> str:
        return f"rank {' / '.join(self.accepted)} in the top {self.k}"


__all__ = ["PolicyRank", "ProbeTask", "ValueSign"]
