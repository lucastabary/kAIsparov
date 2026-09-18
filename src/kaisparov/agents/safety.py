"""Optional blunder filter shared by every agent.

When an agent is built with ``avoid_king_suicide=True`` it discards any candidate
move after which the opponent has mate in one — unless *every* move does, in which
case the position is lost regardless and the full list is returned so the agent
still moves.

Under the old capture-the-king variant this guard dropped the moves that left your
own king en prise, which the rules made legal and which turned "rush the enemy king"
into a free winning strategy in self-play. Standard chess makes those moves illegal
outright, so the guard moved up one rung: the cheapest blunder a baseline can still
make is walking into mate.

This is a policy-level guard, not reward shaping. Torch-free, so the
dependency-light baselines (``RandomAgent``, ``MaterialAgent``) can use it without
pulling in torch.
"""

from __future__ import annotations

from kaisparov.agents.base import Move
from kaisparov.core.game import ChessGame


def walks_into_mate(game: ChessGame, move: Move) -> bool:
    """Whether the opponent has mate in one right after ``move``."""
    undo = game.make(*move)
    try:
        for reply in game.legal_moves():
            reply_undo = game.make(*reply)
            try:
                # is_checkmate() short-circuits on is_check(), so a quiet reply costs
                # one attack lookup rather than a move list.
                if game.is_checkmate():
                    return True
            finally:
                game.unmake(reply_undo)
        return False
    finally:
        game.unmake(undo)


def safe_moves(game: ChessGame, moves: list[Move]) -> list[Move]:
    """Filter ``moves`` down to those that do not walk into mate.

    Falls back to the full ``moves`` list when no safe move exists (the game is lost
    whatever we play), so callers always get at least the moves they passed in.
    """
    safe = [move for move in moves if not walks_into_mate(game, move)]
    return safe or moves


__all__ = ["walks_into_mate", "safe_moves"]
