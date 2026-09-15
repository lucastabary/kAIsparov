"""Tasks: what a contestant is asked to do in a problem's position, and how it is scored.

A :class:`Task` is the polymorphic half of a :class:`~kaisparov.bench.problem.Problem`
(the other half is the position). The runner never looks inside one: it hands the task
a fresh game and a policy, and gets an :class:`Outcome` back. Adding a new kind of test
is therefore one subclass here — the generators, suites, runner and reports all work
with it unchanged.

Two families ship with the base:

- **one-decision tasks** (:class:`MoveTask`) ask for a single move and grade it —
  :class:`FindMove` ("play one of these"), :class:`AvoidMoves` ("never play these");
- **play-outs** (:class:`PlayOut`) let the contestant play the position out against an
  opponent and grade the result — "convert this won endgame", "hold this for 30 plies".

Every task serialises to a JSON-friendly dict tagged with its ``kind``; subclasses
register themselves on definition, so :meth:`Task.from_dict` finds them by name.
Moves are stored as UCI strings (``e2e4``), readable in a suite file.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from kaisparov.agents.base import Policy
from kaisparov.bench.position import mirror_move, move_to_uci, uci_to_move
from kaisparov.core.board import ChessGame
from kaisparov.core.draw import DEFAULT_RULES, DrawRules
from kaisparov.core.movegen import Move, all_moves
from kaisparov.envs.chess_env import ChessEnv

# ------------------------------------------------------------------ run context


OpponentFactory = Callable[[str, int], Policy]


@dataclass
class TaskContext:
    """What a task may need from the benchmark run beyond the position and the policy.

    Kept as one object so a new task that needs something new (a judge, a time budget,
    a cache) grows this class instead of every ``attempt`` signature.
    """

    seed: int = 0
    draw_rules: DrawRules | None = DEFAULT_RULES
    opponent_factory: OpponentFactory | None = None
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


# ---------------------------------------------------------------------- outcome


@dataclass(frozen=True)
class Outcome:
    """How one contestant did on one problem.

    ``score`` is in ``0..1`` and is what averages; ``solved`` is the pass/fail a solve
    rate counts. Most tasks make them agree (score 1 <=> solved), but a task may give
    partial credit. ``seconds`` is the time spent inside the contestant's
    ``select_move`` only — not the opponent's, not the grading.
    """

    score: float
    solved: bool
    move: str | None = None  # first move the contestant played, in UCI
    plies: int = 0  # plies the contestant played
    seconds: float = 0.0
    error: str | None = None  # the contestant crashed, passed, or played an illegal move
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, error: str, *, seconds: float = 0.0, plies: int = 0) -> Outcome:
        return cls(score=0.0, solved=False, plies=plies, seconds=seconds, error=error)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"score": self.score, "solved": self.solved}
        if self.move is not None:
            data["move"] = self.move
        if self.plies:
            data["plies"] = self.plies
        data["seconds"] = round(self.seconds, 6)
        if self.error is not None:
            data["error"] = self.error
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


# ------------------------------------------------------------------------- base


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
        params = dict(data)
        kind = params.pop("kind", None)
        if kind not in Task._kinds:
            raise ValueError(f"unknown task kind {kind!r} (known: {sorted(Task._kinds)})")
        return Task._kinds[kind].from_params(params)

    @staticmethod
    def kinds() -> dict[str, type[Task]]:
        return dict(Task._kinds)


def _legal_uci(game: ChessGame) -> set[str]:
    return {move_to_uci(m) for m in all_moves(game.grid, game.turn, game.en_passant_target)}


# ------------------------------------------------------------ one-decision tasks


class MoveTask(Task):
    """A task decided by a single move. Subclasses only :meth:`grade` it."""

    def attempt(self, game: ChessGame, policy: Policy, context: TaskContext) -> Outcome:
        move, seconds, error = ask_move(policy, game)
        if move is None:
            return Outcome.failure(error or "no move", seconds=seconds)
        score, solved, details = self.grade(game, move_to_uci(move))
        return Outcome(
            score=score,
            solved=solved,
            move=move_to_uci(move),
            plies=1,
            seconds=seconds,
            details=details,
        )

    @abstractmethod
    def grade(self, game: ChessGame, move: str) -> tuple[float, bool, dict[str, Any]]:
        """``(score, solved, details)`` for playing the legal ``move`` (UCI) in ``game``."""


@dataclass(frozen=True)
class FindMove(MoveTask):
    """Play one of the ``accepted`` moves. ``partial`` gives credit to near misses."""

    kind: ClassVar[str] = "find_move"

    accepted: tuple[str, ...]
    partial: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.accepted:
            raise ValueError("FindMove needs at least one accepted move")
        # Normalise so equal tasks compare (and serialise) equal.
        object.__setattr__(self, "accepted", tuple(sorted(set(self.accepted))))
        object.__setattr__(self, "partial", tuple(sorted(dict(self.partial).items())))

    @classmethod
    def of(cls, moves: list[Move], partial: dict[Move, float] | None = None) -> FindMove:
        return cls(
            accepted=tuple(move_to_uci(m) for m in moves),
            partial=tuple((move_to_uci(m), credit) for m, credit in (partial or {}).items()),
        )

    def grade(self, game: ChessGame, move: str) -> tuple[float, bool, dict[str, Any]]:
        if move in self.accepted:
            return 1.0, True, {}
        return dict(self.partial).get(move, 0.0), False, {}

    def validate(self, game: ChessGame) -> None:
        legal = _legal_uci(game)
        for move in (*self.accepted, *dict(self.partial)):
            if move not in legal:
                raise ValueError(f"{self.kind}: {move} is not a legal move here")

    def mirrored(self) -> FindMove:
        def flip(text: str) -> str:
            return move_to_uci(mirror_move(uci_to_move(text)))

        return FindMove(
            accepted=tuple(flip(m) for m in self.accepted),
            partial=tuple((flip(m), credit) for m, credit in self.partial),
        )

    def params(self) -> dict[str, Any]:
        data: dict[str, Any] = {"accepted": list(self.accepted)}
        if self.partial:
            data["partial"] = dict(self.partial)
        return data

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> FindMove:
        return cls(
            accepted=tuple(params["accepted"]),
            partial=tuple(dict(params.get("partial", {})).items()),
        )

    def describe(self) -> str:
        return f"find {' / '.join(self.accepted)}"


@dataclass(frozen=True)
class AvoidMoves(MoveTask):
    """Play anything but the ``forbidden`` moves — the blunders a position invites."""

    kind: ClassVar[str] = "avoid_moves"

    forbidden: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.forbidden:
            raise ValueError("AvoidMoves needs at least one forbidden move")
        object.__setattr__(self, "forbidden", tuple(sorted(set(self.forbidden))))

    @classmethod
    def of(cls, moves: list[Move]) -> AvoidMoves:
        return cls(forbidden=tuple(move_to_uci(m) for m in moves))

    def grade(self, game: ChessGame, move: str) -> tuple[float, bool, dict[str, Any]]:
        solved = move not in self.forbidden
        return float(solved), solved, {}

    def validate(self, game: ChessGame) -> None:
        legal = _legal_uci(game)
        for move in self.forbidden:
            if move not in legal:
                raise ValueError(f"{self.kind}: {move} is not a legal move here")
        if legal <= set(self.forbidden):
            raise ValueError(f"{self.kind}: every legal move is forbidden")

    def mirrored(self) -> AvoidMoves:
        return AvoidMoves(
            forbidden=tuple(move_to_uci(mirror_move(uci_to_move(m))) for m in self.forbidden)
        )

    def params(self) -> dict[str, Any]:
        return {"forbidden": list(self.forbidden)}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> AvoidMoves:
        return cls(forbidden=tuple(params["forbidden"]))

    def describe(self) -> str:
        return f"avoid {' / '.join(self.forbidden)}"


# -------------------------------------------------------------------- play-outs


@dataclass(frozen=True)
class PlayOut(Task):
    """Play the position out against ``opponent`` for at most ``max_plies`` plies.

    The contestant plays the side to move. ``goal`` is ``"win"`` (capture the enemy
    king before the ply cap) or ``"not_lose"`` (still have a king when the game ends,
    by a draw rule or the cap). The score is 1 when the goal is met; a won ``"win"``
    play-out also reports how fast, in ``details["plies"]``, so reports can rank
    conversions by speed.
    """

    kind: ClassVar[str] = "play_out"
    GOALS: ClassVar[tuple[str, ...]] = ("win", "not_lose")

    opponent: str = "material"
    max_plies: int = 40
    goal: str = "win"

    def __post_init__(self) -> None:
        if self.goal not in self.GOALS:
            raise ValueError(f"play_out goal must be one of {self.GOALS}, not {self.goal!r}")
        if self.max_plies <= 0:
            raise ValueError("play_out needs max_plies > 0")

    def attempt(self, game: ChessGame, policy: Policy, context: TaskContext) -> Outcome:
        hero = game.turn
        villain = context.opponent(self.opponent)
        env = ChessEnv(max_plies=self.max_plies, draw_rules=context.draw_rules)
        env.reset(game=game)

        first: str | None = None
        seconds, hero_plies = 0.0, 0
        reason = "max_plies"
        while not env.done:
            mover = env.game.turn
            if mover == hero:
                move, spent, error = ask_move(policy, env.game)
                seconds += spent
                if move is None:
                    if error == "no move" and not env.legal_moves():
                        reason = "stalemate"
                        break  # nothing to play is a draw, not the contestant's fault
                    return Outcome.failure(error or "no move", seconds=seconds, plies=hero_plies)
                hero_plies += 1
                first = first or move_to_uci(move)
            else:
                move = villain.select_move(env.game)
                if move is None or not env.game.is_move_valid(*move):
                    reason = "opponent_forfeit"
                    env.winner = hero
                    break
            result = env.step(move)
            if result.done:
                reason = env.end_reason or "max_plies"

        won = env.winner == hero
        lost = env.winner is not None and not won
        solved = won if self.goal == "win" else not lost
        return Outcome(
            score=float(solved),
            solved=solved,
            move=first,
            plies=hero_plies,
            seconds=seconds,
            details={"result": "win" if won else "loss" if lost else "draw", "end": reason},
        )

    def validate(self, game: ChessGame) -> None:
        if not all_moves(game.grid, game.turn, game.en_passant_target):
            raise ValueError(f"{self.kind}: the side to move has no move")

    def params(self) -> dict[str, Any]:
        return {"opponent": self.opponent, "max_plies": self.max_plies, "goal": self.goal}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> PlayOut:
        return cls(
            opponent=str(params.get("opponent", "material")),
            max_plies=int(params.get("max_plies", 40)),
            goal=str(params.get("goal", "win")),
        )

    def describe(self) -> str:
        return f"{self.goal} vs {self.opponent} in {self.max_plies} plies"


__all__ = [
    "AvoidMoves",
    "FindMove",
    "MoveTask",
    "Outcome",
    "PlayOut",
    "Task",
    "TaskContext",
    "ask_move",
]
