"""Robustness: the model compared with itself, no answer key needed.

- ``mirror_consistency`` — the same position with colours swapped must get the
  mirrored move. The RGCN is *not* colour-symmetric by construction (pawn edges are
  typed by colour), so this is a real question, not a tautology;
- ``distractor_invariance`` — an enemy pawn added where it touches nothing (no square
  White can move to or stands on, no line it blocks for either side) must not change
  the move.
"""

from __future__ import annotations

import random

from kaisparov.bench.generators.base import BoardBuilder, SamplingGenerator, is_quiet
from kaisparov.bench.generators.tactics import board_material
from kaisparov.bench.oracle import Oracle
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import SameMove
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player
from kaisparov.core.rules import pawn_attacks


def _middlegame(rng: random.Random, oracle: Oracle) -> BoardBuilder | None:
    board = BoardBuilder.random_board(rng, board_material(rng, 3, 7), board_material(rng, 3, 7))
    if board is None or not is_quiet(board.game(Player.WHITE), oracle):
        return None
    return board


def _moves(game: ChessGame, player: Player) -> set:
    """Every legal move ``player`` has, even when it is not their turn.

    The mirror check compares a position with its colour-swapped twin, so it has to
    ask both sides what they can do.
    """
    return set((game if player == game.turn else game.passed()).legal_moves())


class MirrorConsistencyGenerator(SamplingGenerator):
    name = "mirror_consistency"
    theme = "mirror_consistency"
    description = "Colours swapped, board flipped: the answer must be the mirrored move."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        board = _middlegame(rng, self.oracle)
        if board is None:
            return None
        position = board.position(Player.WHITE)
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=SameMove(variant=position.mirrored(), mirror=True),
        )


class DistractorInvarianceGenerator(SamplingGenerator):
    name = "distractor_invariance"
    theme = "distractor_invariance"
    description = "An irrelevant enemy pawn is added: the move must not change."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        board = _middlegame(rng, self.oracle)
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        white_before, black_before = _moves(game, Player.WHITE), _moves(game, Player.BLACK)
        touched = {m.dest for m in white_before} | {m.source for m in white_before}

        squares = [s for s in board.free_squares(range(2, BOARD_SIZE - 1))]
        rng.shuffle(squares)
        for square in squares:
            if any(target in touched for target in pawn_attacks(square, Player.BLACK)):
                continue  # it would guard (or threaten) something White cares about
            board.place(Player.BLACK, PieceType.PAWN, square)
            variant = board.position(Player.WHITE)
            after = variant.to_game()
            board.remove(square)
            if _moves(after, Player.WHITE) != white_before or not is_quiet(after, self.oracle):
                continue  # it blocks a white line, or is itself capturable
            if not black_before <= _moves(after, Player.BLACK):
                continue  # it blocks a black line
            if len(_moves(after, Player.BLACK) - black_before) > 2:
                continue  # more new black moves than a pawn's own pushes
            return Problem(
                id="",
                theme=self.theme,
                position=position,
                task=SameMove(variant=variant, mirror=False),
                meta={"pawn": square},
            )
        return None


__all__ = ["DistractorInvarianceGenerator", "MirrorConsistencyGenerator"]
