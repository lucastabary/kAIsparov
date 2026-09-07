"""Uniformly-random baseline policy."""

from __future__ import annotations

import random

from kaisparov.agents.base import Move
from kaisparov.agents.safety import safe_moves
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None, avoid_king_suicide: bool = False):
        self._rng = random.Random(seed)
        # When True, never pick a move that leaves our own king capturable next ply
        # (unless every move does). See kaisparov.agents.safety.
        self.avoid_king_suicide = avoid_king_suicide

    def select_move(self, game: ChessGame) -> Move | None:
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        if not moves:
            return None
        if self.avoid_king_suicide:
            moves = safe_moves(game, moves)
        return self._rng.choice(moves)
