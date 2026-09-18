"""Draw rules: when a game ends without a winner.

Four independent rules, each switchable through :class:`DrawRules`:

``repetition``
    The same position (pieces, side to move, castling rights, en-passant target)
    has occurred N times. Counted from the Zobrist history
    :class:`~kaisparov.core.game.ChessGame` maintains.
``no_progress_plies``
    N plies without a capture or a pawn move — the fifty-move rule, in plies.
    Catches the loops that drift instead of repeating a position exactly.
``insufficient_material``
    Neither side has the material to mate: bare kings, king + one minor against a
    bare king, or bishops on a single colour complex.
``stalemate``
    The side to move is not in check and has no legal move.

The rules are pure predicates over the game state; deciding what to *do* with a
draw (end the episode, score it 0.5, …) belongs to the caller —
:class:`~kaisparov.envs.chess_env.ChessEnv` for play and evaluation, the rollouts
for training.

All four are switchable because the training curriculum and the benchmark need to
turn them off: a generated endgame that is "already drawn" by the material rule
would otherwise never start. Standard play uses :data:`DEFAULT_RULES`, which has
all four on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a core import cycle
    from kaisparov.core.game import ChessGame

REPETITION_LIMIT = 3  # threefold repetition
NO_PROGRESS_PLIES = 100  # 50 moves x 2 plies

# Draw reasons, as reported by :func:`draw_reason`.
REPETITION = "repetition"
NO_PROGRESS = "no_progress"
INSUFFICIENT_MATERIAL = "insufficient_material"
STALEMATE = "stalemate"


@dataclass(frozen=True)
class DrawRules:
    """Which draw rules are active. ``0`` / ``False`` disables a rule."""

    repetition: int = REPETITION_LIMIT
    no_progress_plies: int = NO_PROGRESS_PLIES
    insufficient_material: bool = True
    stalemate: bool = True


DEFAULT_RULES = DrawRules()
NO_RULES = DrawRules(
    repetition=0, no_progress_plies=0, insufficient_material=False, stalemate=False
)


def is_insufficient_material(position) -> bool:
    """True if neither side has enough material to deliver mate.

    Takes a :class:`~kaisparov.core.game.ChessGame` or a bare grid, like
    :mod:`kaisparov.core.rules` — a generated position is often still a grid.
    """
    from kaisparov.core.rules import as_board

    return as_board(position).is_insufficient_material()


def is_stalemate(game: ChessGame) -> bool:
    """True if the side to move is not in check and has no legal move."""
    return game.board.is_stalemate()


def draw_reason(game: ChessGame, rules: DrawRules | None = DEFAULT_RULES) -> str | None:
    """Name the rule that makes ``game`` a draw right now, or ``None``.

    Cheapest first: the two counters, then the material test, then stalemate (which
    has to generate moves).
    """
    if rules is None:
        return None
    if rules.repetition and game.repetition_count() >= rules.repetition:
        return REPETITION
    if rules.no_progress_plies and game.halfmove_clock >= rules.no_progress_plies:
        return NO_PROGRESS
    if rules.insufficient_material and game.board.is_insufficient_material():
        return INSUFFICIENT_MATERIAL
    if rules.stalemate and game.board.is_stalemate():
        return STALEMATE
    return None


def is_draw(game: ChessGame, rules: DrawRules | None = DEFAULT_RULES) -> bool:
    return draw_reason(game, rules) is not None


__all__ = [
    "DrawRules",
    "DEFAULT_RULES",
    "NO_RULES",
    "REPETITION",
    "NO_PROGRESS",
    "INSUFFICIENT_MATERIAL",
    "STALEMATE",
    "REPETITION_LIMIT",
    "NO_PROGRESS_PLIES",
    "is_insufficient_material",
    "is_stalemate",
    "draw_reason",
    "is_draw",
]
