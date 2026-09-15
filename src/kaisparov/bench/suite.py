"""Suites: a named, reproducible list of problems — the benchmark's test set.

A suite has two forms:

- a **spec** (YAML, committed under ``config/benchmarks/``): which generators, how
  many problems each, with which parameters, from which seed. Small, reviewable, and
  enough to rebuild the exact same problems anywhere;
- a **frozen suite** (JSONL, one problem per line after a header): the problems
  themselves. Freeze one when a result must stay comparable across code changes — a
  generator fix alters what a spec produces, a JSONL file never changes.

Every entry draws from its own random stream, seeded from the suite seed and the
entry's name, so adding an entry never reshuffles the problems of the others.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from kaisparov.bench.generators.base import ProblemGenerator
from kaisparov.bench.problem import Problem

FORMAT = "kaisparov-bench-suite/1"


@dataclass(frozen=True)
class SuiteEntry:
    """One ``problems:`` item of a spec: a generator, a count and its parameters."""

    generator: str
    count: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    name: str | None = None  # id prefix; needed when two entries share a generator

    @property
    def label(self) -> str:
        return self.name or self.generator

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuiteEntry:
        unknown = set(data) - {"generator", "count", "params", "name"}
        if unknown:
            raise ValueError(f"unknown suite entry keys: {sorted(unknown)}")
        count = data.get("count")
        return cls(
            generator=str(data["generator"]),
            count=None if count is None else int(count),
            params=dict(data.get("params") or {}),
            name=data.get("name"),
        )


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    entries: tuple[SuiteEntry, ...]
    seed: int = 0
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuiteSpec:
        entries = tuple(SuiteEntry.from_dict(item) for item in data.get("problems") or [])
        if not entries:
            raise ValueError("a suite spec needs at least one entry under 'problems'")
        labels = Counter(entry.label for entry in entries)
        clashes = sorted(label for label, n in labels.items() if n > 1)
        if clashes:
            raise ValueError(f"entries share a name, set 'name:' to tell them apart: {clashes}")
        return cls(
            name=str(data.get("name", "suite")),
            entries=entries,
            seed=int(data.get("seed", 0)),
            description=str(data.get("description", "")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SuiteSpec:
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        data.setdefault("name", Path(path).stem)
        return cls.from_dict(data)

    def build(self) -> Suite:
        problems: list[Problem] = []
        for entry in self.entries:
            # A string seed goes through SHA-512 in `random`: stable across processes
            # and Python versions, unlike hash().
            rng = random.Random(f"{self.seed}:{entry.label}")
            generator = ProblemGenerator.create(entry.generator, entry.params)
            for i, problem in enumerate(generator.generate(entry.count, rng)):
                problems.append(replace(problem, id=f"{entry.label}-{i:04d}"))
        return Suite(self.name, problems, description=self.description, seed=self.seed)


@dataclass
class Suite:
    name: str
    problems: list[Problem]
    description: str = ""
    seed: int | None = None

    def __post_init__(self) -> None:
        ids = Counter(problem.id for problem in self.problems)
        duplicates = sorted(pid for pid, n in ids.items() if n > 1)
        if duplicates:
            raise ValueError(f"suite {self.name!r} has duplicate problem ids: {duplicates[:5]}")

    def __len__(self) -> int:
        return len(self.problems)

    def __iter__(self):
        return iter(self.problems)

    def themes(self) -> list[str]:
        return sorted({problem.theme for problem in self.problems})

    def select(self, themes: Iterable[str] | None = None, limit: int | None = None) -> Suite:
        """A sub-suite: only ``themes`` (all if ``None``), at most ``limit`` per theme."""
        wanted = None if themes is None else set(themes)
        taken: Counter[str] = Counter()
        kept = []
        for problem in self.problems:
            if wanted is not None and problem.theme not in wanted:
                continue
            if limit is not None and taken[problem.theme] >= limit:
                continue
            taken[problem.theme] += 1
            kept.append(problem)
        return Suite(self.name, kept, description=self.description, seed=self.seed)

    # ------------------------------------------------------------------ files
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "format": FORMAT,
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "problems": len(self.problems),
        }
        with path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(header, ensure_ascii=False) + "\n")
            for problem in self.problems:
                stream.write(json.dumps(problem.to_dict(), ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        """Read a frozen ``.jsonl`` suite, or build one from a ``.yaml`` spec."""
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            return SuiteSpec.from_yaml(path).build()
        with path.open("r", encoding="utf-8") as stream:
            lines = [line for line in stream if line.strip()]
        if not lines:
            raise ValueError(f"{path} is empty")
        header = json.loads(lines[0])
        if header.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        problems = [Problem.from_dict(json.loads(line)) for line in lines[1:]]
        return cls(
            name=str(header.get("name", path.stem)),
            problems=problems,
            description=str(header.get("description", "")),
            seed=header.get("seed"),
        )


__all__ = ["Suite", "SuiteEntry", "SuiteSpec"]
