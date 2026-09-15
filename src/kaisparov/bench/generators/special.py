"""Special rules: positions where castling or en passant is the *only* saving move.

Both rules are rare in self-play, so a model may never have learned that they exist.
Each problem puts the king in check with every ordinary defence covered, and the
oracle confirms that exactly one move leaves the king safe — the special one.

- **castle**: the king on e1 is checked down the e-file; rooks on the d- and f-files
  cover its steps; castling carries it to safety (castling out of check is legal in
  this variant, which filters nothing for check).
- **en passant**: a black pawn has just double-pushed next to a white pawn and gives
  check to the king; taking it en passant is the only escape. The double push is a
  real setup move, so the en-passant right comes from history.
"""

from __future__ import annotations

import random

from kaisparov.bench.generators.base import (
    HEAVY_TYPES,
    BoardBuilder,
    GenerationError,
    SamplingGenerator,
    near,
)
from kaisparov.bench.oracle import Oracle
from kaisparov.bench.position import Position, move_to_uci
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import FindMove
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player

KINDS = ("castle", "en_passant")


class SpecialRulesGenerator(SamplingGenerator):
    name = "special_rules"
    theme = "special_rules"
    description = "Castling or en passant is the only move that saves the king."

    def __init__(self, kinds: tuple[str, ...] = KINDS, **kwargs):
        kwargs.setdefault("attempts_per_problem", 3000)
        super().__init__(**kwargs)
        self.kinds = tuple(kinds)
        for kind in self.kinds:
            if kind not in KINDS:
                raise GenerationError(f"special_rules: unknown kind {kind!r} (known: {KINDS})")
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        kind = rng.choice(self.kinds)
        position = self._castle(rng) if kind == "castle" else self._en_passant(rng)
        if position is None:
            return None
        game = position.to_game()
        if game.is_in_check(Player.BLACK) or self.oracle.is_over(game):
            return None
        safe = self.oracle.safe_moves(game)
        if len(safe) != 1 or not self._is_special(kind, game, safe[0]):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=FindMove.of(safe),
            tags=(kind,),
            difficulty=1 if kind == "castle" else 2,
        )

    @staticmethod
    def _is_special(kind: str, game, move) -> bool:
        (sc, sr), (dc, dr) = move
        piece = game.grid[sc][sr]
        if kind == "castle":
            return piece is not None and piece.type == PieceType.KING and abs(dc - sc) == 2
        return (
            piece is not None
            and piece.type == PieceType.PAWN
            and (dc, dr) == game.en_passant_target
        )

    def _castle(self, rng: random.Random) -> Position | None:
        board = BoardBuilder(rng)
        king_col = BOARD_SIZE // 2
        board.place(Player.WHITE, PieceType.KING, (king_col, 0))
        rook_col = rng.choice((0, BOARD_SIZE - 1))
        board.place(Player.WHITE, PieceType.ROOK, (rook_col, 0))
        high = range(2, BOARD_SIZE)
        board.place(
            Player.BLACK,
            rng.choice((PieceType.ROOK, PieceType.QUEEN)),
            squares=[(king_col, r) for r in high],
        )
        for col in (king_col - 1, king_col + 1):  # cover the king's steps
            board.place(Player.BLACK, PieceType.ROOK, squares=[(col, r) for r in high])
        if (
            board.place(Player.BLACK, PieceType.KING, rows=range(BOARD_SIZE - 3, BOARD_SIZE))
            is None
        ):
            return None
        board.place_all(
            Player.BLACK, [rng.choice(HEAVY_TYPES) for _ in range(rng.randint(0, 2))], rows=high
        )
        return board.position(Player.WHITE)

    def _en_passant(self, rng: random.Random) -> Position | None:
        board = BoardBuilder(rng)
        king_col = rng.randint(1, BOARD_SIZE - 2)
        king = (king_col, 3)  # rank 4
        board.place(Player.WHITE, PieceType.KING, king)
        pawn_col = king_col + rng.choice((-1, 1))  # a pawn landing here checks the king
        white_col = pawn_col + rng.choice((-1, 1))
        if not 0 <= white_col < BOARD_SIZE or (white_col, 4) == king:
            return None
        board.place(Player.WHITE, PieceType.PAWN, (white_col, 4))
        board.place(Player.BLACK, PieceType.PAWN, (pawn_col, BOARD_SIZE - 2))
        if board.grid[pawn_col][BOARD_SIZE - 3] is not None:
            return None
        # A defender of the landing square, so the king cannot simply take the pawn.
        guards = [(pawn_col + d, 5) for d in (-1, 1) if 0 <= pawn_col + d < BOARD_SIZE]
        board.place(Player.BLACK, PieceType.PAWN, squares=[g for g in guards if g != (pawn_col, 5)])
        if (
            board.place(Player.BLACK, PieceType.KING, rows=range(BOARD_SIZE - 2, BOARD_SIZE))
            is None
        ):
            return None
        zone = [s for s in near(king, 4) if s[1] != 5 or s[0] != pawn_col]
        board.place_all(
            Player.BLACK, [rng.choice(HEAVY_TYPES) for _ in range(rng.randint(2, 4))], squares=zone
        )
        start = board.position(Player.BLACK)
        push = move_to_uci(((pawn_col, BOARD_SIZE - 2), (pawn_col, 4)))
        try:
            return (
                Position(start.fen, (push,))
                if start.to_game().is_move_valid((pawn_col, 6), (pawn_col, 4))
                else None
            )
        except ValueError:
            return None


__all__ = ["SpecialRulesGenerator"]
