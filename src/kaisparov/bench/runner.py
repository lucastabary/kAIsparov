"""The benchmark loop: every contestant, every problem, one outcome each.

Deliberately dumb. It knows nothing about what a problem tests or how a contestant
plays — tasks and contestants carry that — so a new test family or a new kind of
player never touches this file. What it does own is *fairness*: every contestant sees
the same problems, each from a fresh game, with a policy (and opponent) seeded from
the problem itself.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence

from kaisparov.bench.contestants import Contestant
from kaisparov.bench.problem import Problem
from kaisparov.bench.report import BenchmarkReport, ContestantResult, ProblemResult
from kaisparov.bench.suite import Suite
from kaisparov.bench.tasks import Outcome, TaskContext
from kaisparov.core.draw import DEFAULT_RULES, DrawRules

ProgressFn = Callable[[Contestant, Problem, Outcome, int, int], None]


class BenchmarkRunner:
    def __init__(
        self,
        suite: Suite,
        *,
        seed: int = 0,
        draw_rules: DrawRules | None = DEFAULT_RULES,
        progress: ProgressFn | None = None,
    ):
        self.suite = suite
        self.seed = seed
        self.draw_rules = draw_rules
        self.progress = progress

    def run(self, contestants: Sequence[Contestant]) -> BenchmarkReport:
        names = [c.name for c in contestants]
        if len(set(names)) != len(names):
            raise ValueError(f"contestant names must be unique (label them name=spec): {names}")
        return BenchmarkReport(
            suite=self.suite.name,
            contestants=[self.evaluate(contestant) for contestant in contestants],
            seed=self.seed,
            meta={"problems": len(self.suite)},
        )

    def problem_seed(self, problem: Problem) -> int:
        """The seed of everything random in one attempt, fixed by the problem's id.

        Seeding per problem rather than per run makes an outcome independent of what
        ran before it: a sub-suite (``--themes``, ``--limit``) reproduces exactly the
        outcomes the full suite gave, and a seeded opponent in a play-out makes the
        same moves against every contestant for as long as they play the same moves.
        """
        return zlib.crc32(f"{self.seed}:{problem.id}".encode())

    def evaluate(self, contestant: Contestant) -> ContestantResult:
        result = ContestantResult(contestant.name, contestant.spec)
        total = len(self.suite)
        for index, problem in enumerate(self.suite, start=1):
            seed = self.problem_seed(problem)
            policy = contestant.build(seed)
            context = TaskContext(seed=seed, draw_rules=self.draw_rules)
            outcome = problem.task.attempt(problem.position.to_game(), policy, context)
            result.results.append(
                ProblemResult(problem.id, problem.theme, problem.difficulty, outcome)
            )
            if self.progress is not None:
                self.progress(contestant, problem, outcome, index, total)
        return result


__all__ = ["BenchmarkRunner"]
