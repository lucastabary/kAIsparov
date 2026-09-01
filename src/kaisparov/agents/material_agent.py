"""Greedy 1-ply material baseline.

Plays the move that captures the highest-value enemy piece (capturing a king
ends the game and is therefore always preferred); when no capture is available,
plays a random move. Ties are broken randomly.
"""

from __future__ import annotations

import random

from kaisparov.agents.base import Move
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves
from kaisparov.core.utils import get_piece_value


class MaterialAgent:
    name = "material"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_move(self, game: ChessGame) -> Move | None:
        moves = all_moves(game.grid, game.turn)
        if not moves:
            return None

        best_score = -1.0
        best: list[Move] = []
        for source, dest in moves:
            target = game.grid[dest[0]][dest[1]]
            score = get_piece_value(target.type) if target is not None else 0.0
            if score > best_score:
                best_score = score
                best = [(source, dest)]
            elif score == best_score:
                best.append((source, dest))

        return self._rng.choice(best)
