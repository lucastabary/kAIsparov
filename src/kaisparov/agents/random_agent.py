"""Uniformly-random baseline policy."""

from __future__ import annotations

import random

from kaisparov.agents.base import Move
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_move(self, game: ChessGame) -> Move | None:
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        return self._rng.choice(moves) if moves else None
