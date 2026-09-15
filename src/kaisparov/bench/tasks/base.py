"""The task contract: :class:`Task`, the :class:`Outcome` it returns, and its context."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from kaisparov.agents.base import Policy
from kaisparov.bench.position import move_to_uci
from kaisparov.core.board import ChessGame
from kaisparov.core.draw import DEFAULT_RULES, DrawRules
from kaisparov.core.movegen import Move, all_moves

if TYPE_CHECKING:
    from kaisparov.insights import Analyzer

OpponentFactory = Callable[[str, int], Policy]


@dataclass
class TaskContext:
    """What a task may need from the benchmark run beyond the position and the policy.

    Kept as one object so a new task that needs something new (a judge, a time budget,
    a cache) grows this class instead of every ``attempt`` signature.

    ``analyzer`` is the contestant's window on its own reasoning — its value estimate
    and its ranked move list (see :mod:`kaisparov.insights`). Only the probe tasks read
    it, and a contestant without one simply skips them.
    """

    seed: int = 0
    draw_rules: DrawRules | None = DEFAULT_RULES
    opponent_factory: OpponentFactory | None = None
    analyzer: Analyzer | None = None
    _opponents: dict[str, Policy] = field(default_factory=dict, repr=False)

    def opponent(self, spec: str) -> Policy:
        """The policy a play-out faces, built once per spec and reused afterwards."""
        if spec not in self._opponents:
            if self.opponent_factory is None:
                from kaisparov.bench.contestants import Contestant

                self._opponents[spec] = Contestant.parse(spec).build(self.seed)
            else:
                self._opponents[spec] = self.opponent_factory(spec, self.seed)
        return self._opponents[spec]


@dataclass(frozen=True)
class Outcome:
    """How one contestant did on one problem.

    ``score`` is in ``0..1`` and is what averages; ``solved`` is the pass/fail a solve
    rate counts. Most tasks make them agree (score 1 <=> solved), but a task may give
    partial credit. ``seconds`` is the time spent inside the contestant's
    ``select_move`` (or ``analyze``) only — not the opponent's, not the grading.

    ``skipped`` means the problem does not apply to this contestant (a probe of the
    value head for a player that has none): it counts neither as solved nor as failed.
    """

    score: float
    solved: bool
    move: str | None = None  # first move the contestant played, in UCI
    plies: int = 0  # plies the contestant played
    seconds: float = 0.0
    error: str | None = None  # the contestant crashed, passed, or played an illegal move
    skipped: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, error: str, *, seconds: float = 0.0, plies: int = 0) -> Outcome:
        return cls(score=0.0, solved=False, plies=plies, seconds=seconds, error=error)

    @classmethod
    def skip(cls, reason: str) -> Outcome:
        return cls(score=0.0, solved=False, skipped=True, details={"reason": reason})

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"score": self.score, "solved": self.solved}
        if self.move is not None:
            data["move"] = self.move
        if self.plies:
            data["plies"] = self.plies
        data["seconds"] = round(self.seconds, 6)
        if self.error is not None:
            data["error"] = self.error
        if self.skipped:
            data["skipped"] = True
        if self.details:
            data["details"] = self.details
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Outcome:
        return cls(
            score=float(data["score"]),
            solved=bool(data["solved"]),
            move=data.get("move"),
            plies=int(data.get("plies", 0)),
            seconds=float(data.get("seconds", 0.0)),
            error=data.get("error"),
            skipped=bool(data.get("skipped", False)),
            details=dict(data.get("details", {})),
        )


def ask_move(policy: Policy, game: ChessGame) -> tuple[Move | None, float, str | None]:
    """Ask ``policy`` for a move: ``(move, seconds, error)``.

    Never raises. A crash, a pass (``None``) or an illegal move comes back as an error
    string, so one broken contestant fails its problems instead of the whole benchmark.
    The game is handed over as is; a policy that leaves it modified is a bug the
    engine's own make/unmake discipline already guards against.
    """
    start = time.perf_counter()
    try:
        move = policy.select_move(game)
    except Exception as exc:  # a contestant bug must not abort the benchmark
        return None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    if move is None:
        return None, seconds, "no move"
    if not game.is_move_valid(*move):
        return None, seconds, f"illegal move {move_to_uci(move)}"
    return move, seconds, None


def legal_uci(game: ChessGame) -> set[str]:
    return {move_to_uci(m) for m in all_moves(game.grid, game.turn, game.en_passant_target)}


class Task(ABC):
    """Something to do in a position. Subclasses set ``kind`` and register themselves."""

    kind: ClassVar[str] = ""
    _kinds: ClassVar[dict[str, type[Task]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("kind")
        if not kind:
            return  # an abstract intermediate class
        if kind in Task._kinds:
            raise TypeError(f"task kind {kind!r} is already registered")
        Task._kinds[kind] = cls

    # ---------------------------------------------------------------- contract
    @abstractmethod
    def attempt(self, game: ChessGame, policy: Policy, context: TaskContext) -> Outcome:
        """Let ``policy`` try the task from ``game`` (a fresh game it may consume)."""

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """JSON-friendly parameters; :meth:`from_params` must read them back."""

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict[str, Any]) -> Task: ...

    def validate(self, game: ChessGame) -> None:  # noqa: B027 - an optional hook, not abstract
        """Raise ``ValueError`` if the task makes no sense in ``game``. Default: fine."""

    def mirrored(self) -> Task:
        """This task on the colour-flipped board. Tasks naming squares must override."""
        return self

    def describe(self) -> str:
        return self.kind

    # ----------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params()}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Task:
        import kaisparov.bench.tasks  # noqa: F401 - registers the built-in kinds

        params = dict(data)
        kind = params.pop("kind", None)
        if kind not in Task._kinds:
            raise ValueError(f"unknown task kind {kind!r} (known: {sorted(Task._kinds)})")
        return Task._kinds[kind].from_params(params)

    @staticmethod
    def kinds() -> dict[str, type[Task]]:
        return dict(Task._kinds)


__all__ = ["Outcome", "Task", "TaskContext", "ask_move", "legal_uci"]
