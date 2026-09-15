"""Hand-written problems, listed straight in a suite file.

The escape hatch every benchmark needs: a position someone found in a game, a
regression a model once failed, a study too specific to generate. Each entry is a
problem dict as :meth:`Problem.from_dict` reads it, minus the bookkeeping — ``id``
and ``theme`` default from the generator::

    - generator: fixed
      params:
        theme: sanity
        problems:
          - fen: "4k3/8/8/R3K3/8/8/8/7R w - - 0 1"
            task: {kind: find_move, accepted: [a5e5]}
"""

from __future__ import annotations

import random
from typing import Any

from kaisparov.bench.generators.base import GenerationError, ProblemGenerator
from kaisparov.bench.problem import Problem


class FixedGenerator(ProblemGenerator):
    name = "fixed"
    theme = "fixed"
    description = "Hand-written problems listed in the suite file (FEN + task)."

    def __init__(self, problems: list[dict[str, Any]], theme: str | None = None):
        self.entries = list(problems)
        self.default_theme = theme or self.theme

    def generate(self, count: int | None, rng: random.Random) -> list[Problem]:
        if count is not None and count > len(self.entries):
            raise GenerationError(f"fixed: asked for {count}, only {len(self.entries)} listed")
        entries = self.entries if count is None else self.entries[:count]
        problems = []
        for i, entry in enumerate(entries):
            data = {"id": f"{self.name}-{i:04d}", "theme": self.default_theme, **entry}
            data.setdefault("generator", self.name)
            problem = Problem.from_dict(data)
            problem.validate()  # a typo in a FEN or a move should fail here, loudly
            problems.append(problem)
        return problems
