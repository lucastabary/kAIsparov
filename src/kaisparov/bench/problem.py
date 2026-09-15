"""A benchmark problem: a position, a task in it, and the labels reports group by."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from kaisparov.bench.position import Position
from kaisparov.bench.tasks import Task


@dataclass(frozen=True)
class Problem:
    """One test case. Immutable, JSON-serialisable, comparable.

    ``theme`` is the skill being measured (reports aggregate on it); ``difficulty`` is
    a generator-defined level, higher is harder, so a report can plot solve rate
    against it. ``meta`` carries whatever the generator wants to remember (how the
    solution was found, the material, ...); it is kept out of equality.
    """

    id: str
    theme: str
    position: Position
    task: Task
    difficulty: int = 1
    generator: str = ""
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def validate(self) -> None:
        """Raise ``ValueError`` unless the position loads and the task fits it."""
        self.task.validate(self.position.to_game())

    def mirrored(self) -> Problem:
        """The same problem with colours swapped (see :meth:`Position.mirrored`).

        The ``mirrored`` tag toggles, so mirroring twice gives the original back.
        """
        return replace(
            self,
            position=self.position.mirrored(),
            task=self.task.mirrored(),
            tags=tuple(sorted(set(self.tags) ^ {"mirrored"})),
        )

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "theme": self.theme,
            "difficulty": self.difficulty,
            "fen": self.position.fen,
        }
        if self.position.moves:
            data["moves"] = list(self.position.moves)
        data["task"] = self.task.to_dict()
        if self.generator:
            data["generator"] = self.generator
        if self.tags:
            data["tags"] = list(self.tags)
        if self.meta:
            data["meta"] = self.meta
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Problem:
        return cls(
            id=str(data["id"]),
            theme=str(data.get("theme", "")),
            position=Position(str(data["fen"]), tuple(data.get("moves", ()))),
            task=Task.from_dict(data["task"]),
            difficulty=int(data.get("difficulty", 1)),
            generator=str(data.get("generator", "")),
            tags=tuple(data.get("tags", ())),
            meta=dict(data.get("meta", {})),
        )


__all__ = ["Problem"]
