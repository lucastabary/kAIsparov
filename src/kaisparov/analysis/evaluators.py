"""Position evaluators: "how good is this position for the side to move?".

An evaluator returns a scalar in its *own* unit, plus the ``slope`` that maps that
unit onto a winning probability (see
:func:`~kaisparov.analysis.judge.win_probability`). Keeping the slope on the
evaluator is what lets the judge express every threshold in win-probability points
instead of in units that mean something different per backend — a pawn for the
handcrafted evaluators, an arbitrary critic output for the neural one.

``default_lookahead`` is how many extra plies :class:`~kaisparov.analysis.judge.MoveJudge`
should search on top of each candidate move. Material only changes on captures, so
a static material read cannot see a hanging piece — it needs one reply. A trained
critic is meant to judge a position as it stands and costs a forward pass per node,
so it defaults to no search.

This module stays torch-free; the critic-backed evaluator lives in
:mod:`kaisparov.analysis.critic`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player
from kaisparov.core.utils import get_piece_value


@runtime_checkable
class Evaluator(Protocol):
    """Scores a position from the point of view of the side to move."""

    name: str
    slope: float  # logistic slope mapping this evaluator's unit to a win probability
    default_lookahead: int  # extra plies the judge should search on top of a move

    def evaluate(self, game: ChessGame) -> float:
        """Higher is better *for ``game.turn``*."""
        ...


class MaterialEvaluator:
    """Plain material balance in pawns, from the side to move's point of view.

    Kings are excluded: in capture-the-king a missing king means the game is over,
    which the judge handles as a terminal win rather than as ±100 pawns of material.

    ``slope`` is Lichess' win-probability curve (``0.00368208`` per centipawn), so a
    pawn up reads as ~59% winning chances and three pawns up as ~75%.

    Be aware of what pure material cannot do: every quiet move scores the same, so
    in a calm position it rates the whole move list as equally best. Use it as a
    baseline (and in tests); :class:`HeuristicEvaluator` is what the review UI wants.
    """

    name = "material"
    slope = 0.368
    default_lookahead = 1

    def evaluate(self, game: ChessGame) -> float:
        mover = game.turn
        score = 0.0
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                piece = game.grid[col][row]
                if piece is None or piece.type == PieceType.KING:
                    continue
                value = get_piece_value(piece.type)
                score += value if piece.player == mover else -value
        return score


def _centrality(col: int, row: int) -> float:
    """``0.0`` on the rim, ``1.0`` on the four central squares."""
    middle = (BOARD_SIZE - 1) / 2.0
    return (middle - max(abs(col - middle), abs(row - middle))) / middle


_CENTRALITY: tuple[tuple[float, ...], ...] = tuple(
    tuple(_centrality(col, row) for row in range(BOARD_SIZE)) for col in range(BOARD_SIZE)
)

# Weight, in pawns, of a piece standing on a fully central square. Deliberately
# small next to material (a knight is 3.0): the positional term exists to break
# ties between quiet moves, not to outvote winning a piece.
_CENTRE_WEIGHT: dict[PieceType, float] = {
    PieceType.PAWN: 0.10,
    PieceType.KNIGHT: 0.30,
    PieceType.BISHOP: 0.18,
    PieceType.ROOK: 0.08,
    PieceType.QUEEN: 0.10,
    PieceType.KING: -0.25,  # the king is safer off the middle of the board
}

_PAWN_ADVANCE = 0.06  # per rank pushed, so pawns actually want to move forward


class HeuristicEvaluator(MaterialEvaluator):
    """Material plus a light piece-square term — the default behind the move review.

    Same unit and same win-probability curve as :class:`MaterialEvaluator` (the
    positional bonuses are fractions of a pawn), but it discriminates between quiet
    moves, which is what stops a review from labelling twenty different opening moves
    "best". It is a weak evaluator and makes no claim otherwise; it is here so the
    feature works on a fresh clone, with no trained checkpoint and no torch.
    """

    name = "heuristic"

    def evaluate(self, game: ChessGame) -> float:
        mover = game.turn
        score = 0.0
        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                piece = game.grid[col][row]
                if piece is None:
                    continue

                term = _CENTRE_WEIGHT[piece.type] * _CENTRALITY[col][row]
                if piece.type != PieceType.KING:
                    term += get_piece_value(piece.type)
                if piece.type == PieceType.PAWN:
                    # Ranks pushed from that side's own second rank (White moves up).
                    advanced = row - 1 if piece.player == Player.WHITE else BOARD_SIZE - 2 - row
                    term += _PAWN_ADVANCE * max(0, advanced)
                score += term if piece.player == mover else -term
        return score


__all__ = ["Evaluator", "HeuristicEvaluator", "MaterialEvaluator"]
