"""Optional king-safety move filter shared by every agent.

When an agent is built with ``avoid_king_suicide=True`` it discards any candidate
move that would leave its *own* king immediately capturable by the opponent (a
one-ply king hang) — unless *every* move does, in which case the position is lost
regardless and the full list is returned so the agent still moves.

This is a policy-level guard, not reward shaping: it makes baseline / pool
opponents actually defend their king, which removes the "rush the enemy king"
degenerate strategy from self-play — the enemy king is no longer left hanging, so
a blind knight rush stops being a free win. Torch-free, so the dependency-light
baselines (``RandomAgent``, ``MaterialAgent``) can use it without pulling in torch.
"""

from __future__ import annotations

from kaisparov.agents.base import Move
from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import PieceType, Player


def hangs_own_king(game: ChessGame, move: Move) -> bool:
    """Whether, after ``move``, the opponent can immediately capture the mover's king.

    Plays the move via ``make``/``unmake`` (O(1), no clone). In capture-the-king a
    king is takeable next ply exactly when it is attacked, so this reduces to a single
    :func:`~kaisparov.core.rules.is_in_check` on the mover — no opponent move list is
    generated.
    """
    undo = game.make(*move)
    try:
        # A move that captures the enemy king wins outright — the game ends, there is
        # no reply, so it can never hang our own king.
        if undo.captured is not None and undo.captured.type == PieceType.KING:
            return False
        # ``make`` flipped the turn, so the mover is the side NOT to move now.
        mover = Player.WHITE if game.turn == Player.BLACK else Player.BLACK
        return game.is_in_check(mover)
    finally:
        game.unmake(undo)


def safe_moves(game: ChessGame, moves: list[Move]) -> list[Move]:
    """Filter ``moves`` down to those that don't hang the mover's king.

    Falls back to the full ``moves`` list when no safe move exists (the king is lost
    whatever we play), so callers always get at least the moves they passed in.
    """
    safe = [move for move in moves if not hangs_own_king(game, move)]
    return safe or moves


__all__ = ["hangs_own_king", "safe_moves"]
