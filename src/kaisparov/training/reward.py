"""Build a reward function from :class:`RewardSettings`.

The returned callable takes ``(game, undo)`` where ``game`` is the board *after* the
move and ``undo`` is the handle :meth:`ChessGame.make` returned, and gives back the
shaped reward from the mover's point of view. Used by the self-play rollout.
"""

from __future__ import annotations

from collections.abc import Callable

from kaisparov.core.game import ChessGame, Undo
from kaisparov.core.pieces import PieceType
from kaisparov.core.utils import get_piece_value
from kaisparov.training.config import RewardSettings

RewardFn = Callable[[ChessGame, "Undo"], float]


def make_reward_fn(settings: RewardSettings) -> RewardFn:
    def reward_fn(game: ChessGame, undo: Undo) -> float:
        reward = 0.0
        captured = undo.captured

        if captured is not None:
            reward += settings.material * get_piece_value(captured.type)

        # Promotion is a material event with no capture: the pawn is gone and
        # something far better stands in its place. Without this term the agent has
        # no local signal at all for pushing a pawn home — the gain would only ever
        # reach it through the value function, several plies later.
        if undo.move.promotion is not None:
            gained = get_piece_value(undo.move.promotion) - get_piece_value(PieceType.PAWN)
            reward += settings.promotion * gained

        # Checkmate is the win. It is a flat bonus, decoupled from material, so the
        # scale of a win stays controllable on its own.
        if game.is_checkmate():
            reward += settings.checkmate
        elif settings.check and game.is_in_check(game.turn):
            # After a non-terminal move, game.turn is the opponent: a check means the
            # side that just moved is attacking their king.
            reward += settings.check

        reward -= settings.step_penalty
        return reward

    return reward_fn


__all__ = ["RewardFn", "make_reward_fn"]
