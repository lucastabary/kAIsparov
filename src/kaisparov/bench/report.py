"""Benchmark results: raw per-problem outcomes, and the summaries computed from them.

A report stores every :class:`~kaisparov.bench.tasks.Outcome`, never only averages, so
a question nobody thought of when the benchmark ran — "which problems did v3 fail that
v2 solved?", "does it get worse with more pieces?" — is still answerable from the file.
The summaries (:class:`Tally`) are derived on demand.

Solve rates carry a Wilson 95% interval: on 20 problems, 60% and 75% are not a
difference, and a report that hides that invites reading noise as progress.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaisparov.bench.tasks import Outcome

FORMAT = "kaisparov-bench-report/1"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate (well-behaved at 0% and 100%)."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class ProblemResult:
    problem_id: str
    theme: str
    difficulty: int
    outcome: Outcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem_id,
            "theme": self.theme,
            "difficulty": self.difficulty,
            **self.outcome.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemResult:
        return cls(
            problem_id=str(data["problem"]),
            theme=str(data["theme"]),
            difficulty=int(data.get("difficulty", 1)),
            outcome=Outcome.from_dict(data),
        )


@dataclass(frozen=True)
class Tally:
    """Aggregate over any group of results."""

    count: int = 0
    solved: int = 0
    score: float = 0.0  # summed
    seconds: float = 0.0  # summed decision time
    plies: int = 0
    errors: int = 0

    @classmethod
    def of(cls, results: Iterable[ProblemResult]) -> Tally:
        count = solved = plies = errors = 0
        score = seconds = 0.0
        for result in results:
            outcome = result.outcome
            count += 1
            solved += outcome.solved
            score += outcome.score
            seconds += outcome.seconds
            plies += outcome.plies
            errors += outcome.error is not None
        return cls(count, solved, score, seconds, plies, errors)

    @property
    def solve_rate(self) -> float:
        return self.solved / self.count if self.count else 0.0

    @property
    def mean_score(self) -> float:
        return self.score / self.count if self.count else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.solved, self.count)

    @property
    def ms_per_move(self) -> float:
        return 1000.0 * self.seconds / self.plies if self.plies else 0.0


@dataclass
class ContestantResult:
    name: str
    spec: str
    results: list[ProblemResult] = field(default_factory=list)

    def overall(self) -> Tally:
        return Tally.of(self.results)

    def group(self, key: Callable[[ProblemResult], Any]) -> dict[Any, Tally]:
        buckets: dict[Any, list[ProblemResult]] = defaultdict(list)
        for result in self.results:
            buckets[key(result)].append(result)
        return {name: Tally.of(items) for name, items in sorted(buckets.items())}

    def by_theme(self) -> dict[str, Tally]:
        return self.group(lambda result: result.theme)

    def by_difficulty(self) -> dict[int, Tally]:
        return self.group(lambda result: result.difficulty)

    def metrics(self, prefix: str = "bench") -> dict[str, float]:
        """Flat scalars, shaped for :meth:`RunManager.log_eval` / TensorBoard."""
        flat = {f"{prefix}_solve_rate": self.overall().solve_rate}
        for theme, tally in self.by_theme().items():
            flat[f"{prefix}_{theme}_solve_rate"] = tally.solve_rate
            flat[f"{prefix}_{theme}_score"] = tally.mean_score
        return flat

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "spec": self.spec,
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContestantResult:
        return cls(
            name=str(data["name"]),
            spec=str(data.get("spec", data["name"])),
            results=[ProblemResult.from_dict(item) for item in data.get("results", [])],
        )


@dataclass
class BenchmarkReport:
    suite: str
    contestants: list[ContestantResult] = field(default_factory=list)
    seed: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: dict[str, Any] = field(default_factory=dict)

    def themes(self) -> list[str]:
        return sorted({r.theme for c in self.contestants for r in c.results})

    # ----------------------------------------------------------------- display
    def format_table(self) -> str:
        """Themes down, contestants across: solve rate, with the 95% interval below."""
        names = [c.name for c in self.contestants]
        width = max([12, *(len(name) for name in names)])
        theme_width = max([10, *(len(theme) for theme in self.themes())])
        header = f"{'theme':<{theme_width}} {'n':>4} " + " ".join(f"{n:>{width}}" for n in names)
        lines = [header, "-" * len(header)]

        tallies = [c.by_theme() for c in self.contestants]
        rows = [(theme, [t.get(theme, Tally()) for t in tallies]) for theme in self.themes()]
        rows.append(("ALL", [c.overall() for c in self.contestants]))
        for theme, row in rows:
            count = max((t.count for t in row), default=0)
            cells = " ".join(f"{t.solve_rate:>{width}.0%}" for t in row)
            lines.append(f"{theme:<{theme_width}} {count:>4} {cells}")
            spans = " ".join(f"{f'{t.interval[0]:.0%}-{t.interval[1]:.0%}':>{width}}" for t in row)
            lines.append(f"{'':<{theme_width}} {'':>4} {spans}")

        footer = []
        for contestant in self.contestants:
            overall = contestant.overall()
            note = f"{contestant.name}: {overall.ms_per_move:.1f} ms/move"
            if overall.errors:
                note += f", {overall.errors} errors"
            footer.append(note)
        return "\n".join([*lines, "", *footer])

    # ------------------------------------------------------------------- files
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "suite": self.suite,
            "seed": self.seed,
            "created_at": self.created_at,
            "meta": self.meta,
            "contestants": [c.to_dict() for c in self.contestants],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkReport:
        if data.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} report")
        return cls(
            suite=str(data["suite"]),
            contestants=[ContestantResult.from_dict(c) for c in data.get("contestants", [])],
            seed=int(data.get("seed", 0)),
            created_at=str(data.get("created_at", "")),
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkReport:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["BenchmarkReport", "ContestantResult", "ProblemResult", "Tally", "wilson_interval"]
