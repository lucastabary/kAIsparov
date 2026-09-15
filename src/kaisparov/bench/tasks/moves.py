"""One-decision tasks: the contestant plays a single move, and that move is graded."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from kaisparov.agents.base import Policy
from kaisparov.bench.position import mirror_move, move_to_uci, uci_to_move
from kaisparov.bench.tasks.base import Outcome, Task, TaskContext, ask_move, legal_uci
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import Move


def _flip(move: str) -> str:
    return move_to_uci(mirror_move(uci_to_move(move)))


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
        legal = legal_uci(game)
        for move in (*self.accepted, *dict(self.partial)):
            if move not in legal:
                raise ValueError(f"{self.kind}: {move} is not a legal move here")

    def mirrored(self) -> FindMove:
        return FindMove(
            accepted=tuple(_flip(m) for m in self.accepted),
            partial=tuple((_flip(m), credit) for m, credit in self.partial),
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
        legal = legal_uci(game)
        for move in self.forbidden:
            if move not in legal:
                raise ValueError(f"{self.kind}: {move} is not a legal move here")
        if legal <= set(self.forbidden):
            raise ValueError(f"{self.kind}: every legal move is forbidden")

    def mirrored(self) -> AvoidMoves:
        return AvoidMoves(forbidden=tuple(_flip(m) for m in self.forbidden))

    def params(self) -> dict[str, Any]:
        return {"forbidden": list(self.forbidden)}

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> AvoidMoves:
        return cls(forbidden=tuple(params["forbidden"]))

    def describe(self) -> str:
        return f"avoid {' / '.join(self.forbidden)}"


@dataclass(frozen=True)
class WinMaterial(MoveTask):
    """Play a move that wins at least ``min_gain`` pawns against best defence.

    Graded by *searching the move actually played* (:meth:`Oracle.material_gain`,
    ``plies`` replies deep), not by comparing it with a list: any move that wins the
    material is right, including one the generator never enumerated. ``best_gain`` —
    the most any move wins — turns the gain into a score, so winning a pawn where a
    rook was available earns a little credit but does not pass.
    """

    kind: ClassVar[str] = "win_material"

    min_gain: float
    best_gain: float
    plies: int = 3
    reference: tuple[str, ...] = ()  # moves the generator found; informative only

    def __post_init__(self) -> None:
        if self.min_gain <= 0 or self.best_gain < self.min_gain:
            raise ValueError("win_material needs 0 < min_gain <= best_gain")
        object.__setattr__(self, "reference", tuple(sorted(set(self.reference))))

    def grade(self, game: ChessGame, move: str) -> tuple[float, bool, dict[str, Any]]:
        from kaisparov.bench.oracle import Oracle

        gain = Oracle().material_gain(game, uci_to_move(move), self.plies)
        score = max(0.0, min(1.0, gain / self.best_gain))
        return score, gain >= self.min_gain, {"gain": gain}

    def validate(self, game: ChessGame) -> None:
        legal = legal_uci(game)
        for move in self.reference:
            if move not in legal:
                raise ValueError(f"{self.kind}: {move} is not a legal move here")

    def mirrored(self) -> WinMaterial:
        return WinMaterial(
            self.min_gain, self.best_gain, self.plies, tuple(_flip(m) for m in self.reference)
        )

    def params(self) -> dict[str, Any]:
        return {
            "min_gain": self.min_gain,
            "best_gain": self.best_gain,
            "plies": self.plies,
            "reference": list(self.reference),
        }

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> WinMaterial:
        return cls(
            min_gain=float(params["min_gain"]),
            best_gain=float(params["best_gain"]),
            plies=int(params.get("plies", 3)),
            reference=tuple(params.get("reference", ())),
        )

    def describe(self) -> str:
        return f"win {self.min_gain:g}+ pawns (best {self.best_gain:g})"


__all__ = ["AvoidMoves", "FindMove", "MoveTask", "WinMaterial"]
