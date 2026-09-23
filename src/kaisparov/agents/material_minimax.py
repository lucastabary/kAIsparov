"""Alpha-beta search on material: a baseline that looks ahead.

``MaterialAgent`` grabs the biggest capture on the board and nothing else, so it walks
into every recapture. This one searches ``depth`` plies (negamax, alpha-beta) and
scores each leaf by the material balance: it keeps its pieces defended, refuses a
poisoned capture, takes a hung piece, and sees a mate within its horizon. Still no
model and no torch — a sparring partner that punishes the learner's material blunders
without any shaping in the learner's own reward.

Material is tracked move by move (:func:`~kaisparov.core.material.move_gain`) rather
than recounted from the board at each leaf, which keeps a depth-2 search affordable in
the training rollouts. As every search here must (see CLAUDE.md), a node with no legal
move is scored as mate (``-WIN``) or stalemate (0), and so is any drawn position: a
side up a queen does not "win" a stalemate.
"""

from __future__ import annotations

import random

from kaisparov.agents.base import Move
from kaisparov.agents.safety import safe_moves
from kaisparov.core.draw import DEFAULT_RULES, DrawRules
from kaisparov.core.game import ChessGame
from kaisparov.core.material import WIN, gain_if_played, material_balance, move_gain


class MaterialMinimaxAgent:
    name = "material_minimax"

    def __init__(
        self,
        depth: int = 2,
        seed: int | None = None,
        avoid_king_suicide: bool = False,
        draw_rules: DrawRules | None = DEFAULT_RULES,
    ):
        if depth < 1:
            raise ValueError(f"MaterialMinimaxAgent needs depth >= 1, got {depth}")
        self.depth = depth
        self._rng = random.Random(seed)
        # At depth 1 the search does not see the opponent's mating reply; this guard
        # drops root moves that walk into mate in one. Depth >= 2 sees it on its own.
        self.avoid_king_suicide = avoid_king_suicide
        self.draw_rules = draw_rules

    def _ordered(self, game: ChessGame, moves: list[Move]) -> list[Move]:
        """Captures and promotions first, biggest first: they are what prunes."""
        return sorted(moves, key=lambda move: gain_if_played(game, move), reverse=True)

    def _search(
        self, game: ChessGame, depth: int, alpha: float, beta: float, balance: float, ply: int
    ) -> float:
        """Negamax value for the side to move, whose material lead is ``balance``."""
        if game.is_checkmate():
            return -(WIN - ply)  # mated: the later, the less bad
        if game.is_draw(self.draw_rules):
            return 0.0  # stalemate, repetition, dead position: a draw, whatever the count
        if depth == 0:
            return balance
        best = -2 * WIN
        for move in self._ordered(game, game.legal_moves()):
            undo = game.make(*move)
            value = -self._search(
                game, depth - 1, -beta, -alpha, -(balance + move_gain(undo)), ply + 1
            )
            game.unmake(undo)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def select_move(self, game: ChessGame) -> Move | None:
        moves = game.legal_moves()
        if not moves:
            return None
        if self.avoid_king_suicide:
            moves = safe_moves(game, moves)

        balance = material_balance(game, game.turn)
        best_value = -2 * WIN
        best: list[Move] = []
        for move in self._ordered(game, moves):
            undo = game.make(*move)
            # The window stays just below the best value found, so every move that
            # ties it gets an exact score and a fair share of the random tie-break.
            value = -self._search(
                game,
                self.depth - 1,
                -2 * WIN,
                -(best_value - 1e-9),
                -(balance + move_gain(undo)),
                1,
            )
            game.unmake(undo)
            if value > best_value + 1e-9:
                best_value, best = value, [move]
            elif value >= best_value - 1e-9:
                best.append(move)
        return self._rng.choice(best)


__all__ = ["MaterialMinimaxAgent"]
