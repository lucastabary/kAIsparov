"""Sanity check: the enemy king is en prise — take it.

The simplest problem the variant has, and the one every other result depends on: a
model that walks past a free king capture is not ready for anything subtler. Random
sparse boards are drawn until the side to move attacks the enemy king; the accepted
answers are exactly the king captures, as the oracle lists them.

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


class KingCaptureGenerator(SamplingGenerator):
    name = "king_capture"
    theme = "king_capture"
    description = "The enemy king is en prise: capture it (sanity check)."

    def __init__(self, min_pieces: int = 1, max_pieces: int = 5, **kwargs):
        super().__init__(**kwargs)
        if not 1 <= min_pieces <= max_pieces:
            raise ValueError("king_capture needs 1 <= min_pieces <= max_pieces")
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
        if not board.in_check(Player.BLACK):
            return None

        position = board.position(Player.WHITE)
        captures = self.oracle.king_captures(position.to_game())
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=FindMove.of(captures),
            difficulty=white + black,  # more pieces, more distractions
            meta={"captures": len(captures)},
        )
