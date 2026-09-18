"""Sanity check: there is mate in one — play it.

The simplest problem the benchmark has, and the one every other result leans on: a
model that walks past a mate in one is not ready for anything subtler. Random sparse
boards are drawn until the side to move has a mating move; the accepted answers are
exactly those moves, as the oracle lists them.

It doubles as the reference implementation of a
:class:`~kaisparov.bench.generators.base.SamplingGenerator`.
"""

from __future__ import annotations

import random

from kaisparov.bench.generators.base import BoardBuilder, SamplingGenerator
from kaisparov.bench.oracle import Oracle
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import FindMove
from kaisparov.core.pieces import PieceType, Player


class MateInOneGenerator(SamplingGenerator):
    name = "mate_in_one"
    theme = "mate_in_one"
    description = "There is mate in one: find it (sanity check)."

    def __init__(self, min_pieces: int = 2, max_pieces: int = 6, **kwargs):
        super().__init__(**kwargs)
        if not 1 <= min_pieces <= max_pieces:
            raise ValueError("mate_in_one needs 1 <= min_pieces <= max_pieces")
        self.min_pieces = min_pieces
        self.max_pieces = max_pieces
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        board = BoardBuilder(rng)
        white = rng.randint(self.min_pieces, self.max_pieces)
        black = rng.randint(0, self.max_pieces)
        # Kings go down first so the others fill in around them; neither placement can
        # fail on a board this sparse.
        board.place(Player.BLACK, PieceType.KING)
        board.place(Player.WHITE, PieceType.KING)
        if not (
            board.place_all(Player.WHITE, board.random_material(white))
            and board.place_all(Player.BLACK, board.random_material(black))
        ):
            return None
        if board.in_check(Player.BLACK):
            return None  # Black is already in check with White to move: not a position

        position = board.position(Player.WHITE)
        mates = self.oracle.mates_in_one(position.to_game())
        if not mates:
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=FindMove.of(mates),
            difficulty=white + black,  # more pieces, more distractions
            meta={"mates": len(mates)},
        )
