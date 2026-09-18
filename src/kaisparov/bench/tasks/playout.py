"""Play-outs: the contestant plays the position out against an opponent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from kaisparov.agents.base import Policy
from kaisparov.bench.position import move_to_uci
from kaisparov.bench.tasks.base import Outcome, Task, TaskContext, ask_move
from kaisparov.core.game import ChessGame
from kaisparov.envs.chess_env import ChessEnv


@dataclass(frozen=True)
class PlayOut(Task):
    """Play the position out against ``opponent`` for at most ``max_plies`` plies.

    The contestant plays the side to move. ``goal`` is ``"win"`` (capture the enemy
    king before the ply cap) or ``"not_lose"`` (still have a king when the game ends,
    by a draw rule or the cap). The score is 1 when the goal is met; the outcome's
    ``plies`` says how many moves it took, so reports can rank conversions by speed.
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
            if env.game.turn == hero:
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
        if not game.legal_moves():
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


__all__ = ["PlayOut"]
