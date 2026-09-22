"""Greedy 1-ply material baseline.

Plays the move that wins the most material: the highest-value capture, or the
material a promotion gains, whichever is larger. With nothing to win it plays a
random move. Ties are broken randomly.

It is deliberately blind to everything else — it will happily take a poisoned piece.
That is what makes it a baseline; ``avoid_king_suicide`` bolts on the one guard that
keeps it from losing outright (see :mod:`kaisparov.agents.safety`).
"""

from __future__ import annotations

import random

from kaisparov.agents.base import Move
from kaisparov.agents.safety import safe_moves
from kaisparov.core.game import ChessGame
from kaisparov.core.material import gain_if_played


class MaterialAgent:
    name = "material"

    def __init__(self, seed: int | None = None, avoid_king_suicide: bool = False):
        self._rng = random.Random(seed)
        # When True, drop the moves that walk into mate in one before the greedy
        # capture pick. See kaisparov.agents.safety.
        self.avoid_king_suicide = avoid_king_suicide

    def select_move(self, game: ChessGame) -> Move | None:
        moves = game.legal_moves()
        if not moves:
            return None
        if self.avoid_king_suicide:
            moves = safe_moves(game, moves)

        best_score = -1.0
        best: list[Move] = []
        for move in moves:
            # Asked of the engine rather than read off the destination square, which
            # is empty on an en passant capture. Underpromotions gain less than a
            # queen, so the greedy pick queens unless a capture beats it.
            score = gain_if_played(game, move)
            if score > best_score:
                best_score = score
                best = [move]
            elif score == best_score:
                best.append(move)

        return self._rng.choice(best)
