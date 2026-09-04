"""Build a reward function from :class:`RewardSettings`.

The returned callable takes ``(game, captured)`` where ``game`` is the board *after*
the move and ``captured`` is the piece removed by it (or ``None``), and returns the
shaped reward from the mover's point of view. Used by the self-play rollout.
"""

from __future__ import annotations

from collections.abc import Callable

from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.core.utils import get_piece_value
from kaisparov.training.config import RewardSettings

RewardFn = Callable[[ChessGame, "Piece | None"], float]


def make_reward_fn(settings: RewardSettings) -> RewardFn:
    def reward_fn(game: ChessGame, captured: Piece | None) -> float:
        reward = 0.0
        king_captured = captured is not None and captured.type == PieceType.KING

        if king_captured:
            # A win is the flat ``king_capture`` bonus, decoupled from the king's
            # (sentinel) material value so the win reward stays controllable.
            reward += settings.king_capture
        elif captured is not None:
            reward += settings.material * get_piece_value(captured.type)

        # After a non-terminal move, game.turn is the opponent; a check means the
        # opponent's king is now attacked by the side that just moved.
        if settings.check and not king_captured and game.is_in_check(game.turn):
            reward += settings.check

        # King safety: penalise leaving your OWN king capturable (dense signal that
        # directly discourages hanging the king). The mover is the side NOT to move
        # now, since ``game.turn`` already flipped to the opponent.
        if settings.king_safety and not king_captured:
            mover = Player.WHITE if game.turn == Player.BLACK else Player.BLACK
            if game.is_in_check(mover):
                reward -= settings.king_safety

        reward -= settings.step_penalty
        return reward

    return reward_fn
