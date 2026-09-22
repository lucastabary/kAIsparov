"""Material accounting, in pawns — the one place that says what pieces and moves are worth.

Everyone who counts material (the env's reward, the reward shaping, the greedy
baseline, the evaluators, the move judge, the benchmark oracle) goes through here, so
a promotion or an en passant capture is valued the same way everywhere.

Kings are not material: a game is won by mate, which searches score as ``±WIN``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaisparov.core.pieces import PieceType, Player
from kaisparov.core.utils import get_piece_value

if TYPE_CHECKING:
    from kaisparov.core.game import ChessGame, Undo

# The score of a won position, in pawns: far above any material count, so a search
# prefers any mate to any material, and still a plain float.
WIN = 1e6


def promotion_gain(piece_type: PieceType) -> float:
    """What promoting to ``piece_type`` gains: the new piece, minus the pawn it was."""
    return get_piece_value(piece_type) - get_piece_value(PieceType.PAWN)


def captured_value(undo: Undo) -> float:
    """The value of the piece the move took (en passant included), or 0."""
    return get_piece_value(undo.captured.type) if undo.captured is not None else 0.0


def move_gain(undo: Undo) -> float:
    """The material a played move won for its mover: what it took, plus any promotion."""
    gain = captured_value(undo)
    if undo.move.promotion is not None:
        gain += promotion_gain(undo.move.promotion)
    return gain


def material_balance(game: ChessGame, player: Player) -> float:
    """``player``'s material minus the opponent's, kings excluded."""
    grid = game.grid
    score = 0.0
    for column in grid:
        for piece in column:
            if piece is None or piece.type == PieceType.KING:
                continue
            value = get_piece_value(piece.type)
            score += value if piece.player == player else -value
    return score


__all__ = ["WIN", "captured_value", "material_balance", "move_gain", "promotion_gain"]
