"""Draw rules: when a game ends with nobody capturing a king.

Three independent rules, each switchable through :class:`DrawRules`:

``repetition``
    The same position (pieces, side to move, castling rights, en-passant target)
    has occurred N times — the players are shuffling in a loop. Counted from the
    per-position tally :class:`~kaisparov.core.board.ChessGame` maintains.
``no_progress_plies``
    N plies without a capture or a pawn move — the fifty-move rule, in plies.
    Catches the loops that drift instead of repeating a position exactly.
``insufficient_material``
    Neither side has the material to mate: bare kings, king + one minor against a
    bare king, or one bishop each on the same colour complex.

.. warning::
   The material rule is a **convention** in this variant, not a fact. Moves here
   are pseudo-legal and the game ends on a king capture, so a king may walk next
   to the enemy king and be taken: strictly speaking, no position is ever dead —
   even bare kings can be "won" on a blunder. The rule declares those positions
   drawn because nothing can be *played for* in them, which stops blunder
   roulette from deciding endgames. Turn it off (``insufficient_material=False``)
   to keep them alive.

The rules are pure predicates over the game state; deciding what to *do* with a
draw (end the episode, score it 0.5, …) belongs to the caller —
:class:`~kaisparov.envs.chess_env.ChessEnv` for play and evaluation, the rollouts
for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kaisparov.core.coords import ALL_SQUARES
from kaisparov.core.pieces import Piece, PieceType, Player

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a core import cycle
    from kaisparov.core.board import ChessGame

Grid = list[list["Piece | None"]]

REPETITION_LIMIT = 3  # threefold repetition
NO_PROGRESS_PLIES = 100  # 50 moves x 2 plies

# Draw reasons, as reported by :func:`draw_reason`.
REPETITION = "repetition"
NO_PROGRESS = "no_progress"
INSUFFICIENT_MATERIAL = "insufficient_material"

_MINORS = (PieceType.BISHOP, PieceType.KNIGHT)


@dataclass(frozen=True)
class DrawRules:
    """Which draw rules are active. ``0`` / ``False`` disables a rule."""

    repetition: int = REPETITION_LIMIT
    no_progress_plies: int = NO_PROGRESS_PLIES
    insufficient_material: bool = True


DEFAULT_RULES = DrawRules()
NO_RULES = DrawRules(repetition=0, no_progress_plies=0, insufficient_material=False)


def is_insufficient_material(grid: Grid) -> bool:
    """True if neither side has enough material to mate (see the module warning)."""
    minors = {Player.WHITE: 0, Player.BLACK: 0}
    bishop_squares: list[int] = []

    for x, y in ALL_SQUARES:
        piece = grid[x][y]
        if piece is None or piece.type == PieceType.KING:
            continue
        if piece.type not in _MINORS:
            return False  # a pawn, rook or queen is always enough
        minors[piece.player] += 1
        if minors[piece.player] > 1:
            return False  # two minors can mate
        if piece.type == PieceType.BISHOP:
            bishop_squares.append((x + y) % 2)

    if minors[Player.WHITE] + minors[Player.BLACK] <= 1:
        return True  # K vs K, or K + one minor vs K
    # One minor each: only two bishops on the same colour complex are dead.
    return len(bishop_squares) == 2 and bishop_squares[0] == bishop_squares[1]


def draw_reason(game: ChessGame, rules: DrawRules | None = DEFAULT_RULES) -> str | None:
    """Name the rule that makes ``game`` a draw right now, or ``None``.

    The two O(1) counters are tested before the material scan, so the common
    "not a draw" answer costs two integer comparisons.
    """
    if rules is None:
        return None
    if rules.repetition and game.repetition_count() >= rules.repetition:
        return REPETITION
    if rules.no_progress_plies and game.halfmove_clock >= rules.no_progress_plies:
        return NO_PROGRESS
    if rules.insufficient_material and is_insufficient_material(game.grid):
        return INSUFFICIENT_MATERIAL
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
    "REPETITION_LIMIT",
    "NO_PROGRESS_PLIES",
    "is_insufficient_material",
    "draw_reason",
    "is_draw",
]
