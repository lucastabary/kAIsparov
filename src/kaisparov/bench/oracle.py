"""Ground truth for generated problems: exact, exhaustive, and slow on purpose.

A generator proposes a position; the :class:`Oracle` decides whether it is a problem
and what its answer is. It must be *right*, not fast — a benchmark whose answer key is
wrong measures nothing — so it searches every line and applies the same end-of-game
rules as :class:`~kaisparov.envs.chess_env.ChessEnv`, draw rules included. That last
part matters more than it looks: capturing the last defender may leave an
"insufficient material" draw, and boxing the king in may be a stalemate. A search that
ignored them would call both a win, and a model that avoids them would be marked wrong.

The central question is "can the side to move force a king capture within ``depth``
of its own moves, whatever the defence?" — the capture-the-king analogue of
mate-in-N (depth 1 is "the king is en prise", depth 2 "mate in one" in classical
terms). It is an AND/OR search, cut at the first refutation, and uses one fact of the
variant to stay affordable: the side to move can take the king *right now* exactly
when that king is attacked, so the last ply is one attack lookup, never a move list.

Torch-free, and it leaves the game exactly as it found it.
"""

from __future__ import annotations

from kaisparov.bench.position import other
from kaisparov.core.board import ChessGame
from kaisparov.core.draw import DEFAULT_RULES, DrawRules, draw_reason
from kaisparov.core.movegen import Move, all_moves
from kaisparov.core.pieces import PieceType


class Oracle:
    def __init__(self, draw_rules: DrawRules | None = DEFAULT_RULES):
        self.draw_rules = draw_rules

    # ---------------------------------------------------------------- queries
    def winning_moves(self, game: ChessGame, depth: int) -> list[Move]:
        """Every move that forces a king capture within ``depth`` of the mover's moves."""
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        return [move for move in moves if self._move_wins(game, move, depth)]

    def wins_within(self, game: ChessGame, depth: int) -> bool:
        """Whether the side to move can force a king capture within ``depth`` moves."""
        return self._attacker_wins(game, depth)

    def win_depth(self, game: ChessGame, max_depth: int) -> int | None:
        """The fewest moves in which the side to move forces a win, or ``None``."""
        for depth in range(1, max_depth + 1):
            if self._attacker_wins(game, depth):
                return depth
        return None

    def king_captures(self, game: ChessGame) -> list[Move]:
        """The moves that take the enemy king on the spot."""
        return [
            move
            for move in all_moves(game.grid, game.turn, game.en_passant_target)
            if (target := game.grid[move[1][0]][move[1][1]]) is not None
            and target.type == PieceType.KING
        ]

    # ----------------------------------------------------------------- search
    def _ended_in_draw(self, game: ChessGame) -> bool:
        """The env's end-of-step test, minus the king capture the caller has handled."""
        if not all_moves(game.grid, game.turn, game.en_passant_target):
            return True
        return draw_reason(game, self.draw_rules) is not None

    def _attacker_wins(self, game: ChessGame, depth: int) -> bool:
        if game.is_in_check(other(game.turn)):
            return True  # the enemy king is en prise: take it
        if depth <= 1:
            return False
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        return any(self._move_wins(game, move, depth) for move in moves)

    def _move_wins(self, game: ChessGame, move: Move, depth: int) -> bool:
        mover = game.turn
        undo = game.make(*move)
        try:
            if undo.captured is not None and undo.captured.type == PieceType.KING:
                return True
            if depth <= 1:
                return False
            if game.is_in_check(mover):
                return False  # hangs the king: the defender takes it first
            if self._ended_in_draw(game):
                return False
            return self._defender_loses(game, depth - 1)
        finally:
            game.unmake(undo)

    def _defender_loses(self, game: ChessGame, depth: int) -> bool:
        """Every defence (the side to move's) still allows a win within ``depth``."""
        for reply in all_moves(game.grid, game.turn, game.en_passant_target):
            undo = game.make(*reply)
            try:
                if undo.captured is not None and undo.captured.type == PieceType.KING:
                    return False
                if self._ended_in_draw(game) or not self._attacker_wins(game, depth):
                    return False
            finally:
                game.unmake(undo)
        return True


__all__ = ["Oracle"]
