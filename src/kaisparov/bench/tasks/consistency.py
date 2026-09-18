"""Consistency tasks: the answer is not right or wrong, it has to *agree with itself*.

A model that understands a position plays the same move when nothing relevant changed
— a pawn added far from the action — and the mirrored move when the colours are
swapped. Neither needs an answer key: the contestant is asked twice and compared with
itself, so these tests measure robustness independently of strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from kaisparov.agents.base import Policy
from kaisparov.bench.position import Position, mirror_move, move_to_uci
from kaisparov.bench.tasks.base import Outcome, Task, TaskContext, ask_move
from kaisparov.core.game import ChessGame


@dataclass(frozen=True)
class SameMove(Task):
    """Play the problem's position, then ``variant``: the two answers must match.

    With ``mirror`` the variant is the colour-flipped board and "match" means the
    mirrored move; otherwise it means the very same move.
    """

    kind: ClassVar[str] = "same_move"

    variant: Position
    mirror: bool = False

    def attempt(self, game: ChessGame, policy: Policy, context: TaskContext) -> Outcome:
        first, seconds, error = ask_move(policy, game)
        if first is None:
            return Outcome.failure(error or "no move", seconds=seconds)
        second, more, error = ask_move(policy, self.variant.to_game())
        seconds += more
        if second is None:
            return Outcome.failure(error or "no move", seconds=seconds, plies=1)
        expected = mirror_move(first) if self.mirror else first
        solved = second == expected
        return Outcome(
            score=float(solved),
            solved=solved,
            move=move_to_uci(first),
            plies=2,
            seconds=seconds,
            details={"variant_move": move_to_uci(second)},
        )

    def validate(self, game: ChessGame) -> None:
        variant = self.variant.to_game()
        if (variant.turn != game.turn) != self.mirror:
            raise ValueError(f"{self.kind}: the variant's side to move does not fit")

    def mirrored(self) -> SameMove:
        return SameMove(self.variant.mirrored(), self.mirror)

    def params(self) -> dict[str, Any]:
        data: dict[str, Any] = {"variant": self.variant.fen, "mirror": self.mirror}
        if self.variant.moves:
            data["variant_moves"] = list(self.variant.moves)
        return data

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SameMove:
        variant = Position(str(params["variant"]), tuple(params.get("variant_moves", ())))
        return cls(variant=variant, mirror=bool(params.get("mirror", False)))

    def describe(self) -> str:
        return "mirrored move" if self.mirror else "same move in the variant"


__all__ = ["SameMove"]
