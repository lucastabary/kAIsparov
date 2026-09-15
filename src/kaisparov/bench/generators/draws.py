"""Draw traps: winning positions where some moves throw the win away into a draw.

A draw is never a reward event in training (see CLAUDE.md), so nothing in the reward
teaches a model to avoid one — only the lost *chance* to win does, very indirectly.
These themes measure whether that lesson got through:

- ``stalemate_trap`` — heavy pieces against a bare king; some moves stalemate it;
- ``repetition_trap`` — a won position reached by shuffling; one move repeats it a
  third time. The shuffle is real history (setup moves), not a FEN annotation.
"""

from __future__ import annotations

import random

from kaisparov.bench.generators.base import EDGE_SQUARES, BoardBuilder, SamplingGenerator, is_quiet
from kaisparov.bench.oracle import Oracle
from kaisparov.bench.position import Position, move_to_uci
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import AvoidMoves
from kaisparov.core.board import ChessGame
from kaisparov.core.draw import REPETITION, STALEMATE
from kaisparov.core.movegen import Move, all_moves
from kaisparov.core.pieces import PieceType, Player

_ARMIES = {
    "Q": [PieceType.QUEEN],
    "R": [PieceType.ROOK],
    "QR": [PieceType.QUEEN, PieceType.ROOK],
    "RR": [PieceType.ROOK, PieceType.ROOK],
    "QQ": [PieceType.QUEEN, PieceType.QUEEN],
}


def army(name: str) -> list[PieceType]:
    if name not in _ARMIES:
        raise ValueError(f"unknown material {name!r} (known: {sorted(_ARMIES)})")
    return list(_ARMIES[name])


class StalemateTrapGenerator(SamplingGenerator):
    name = "stalemate_trap"
    theme = "stalemate_trap"
    description = "Heavy pieces against a bare king: avoid the moves that stalemate it."

    def __init__(self, materials: tuple[str, ...] = ("Q", "QR", "RR"), **kwargs):
        super().__init__(**kwargs)
        self.materials = tuple(materials)
        for material in self.materials:
            army(material)
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        material = rng.choice(self.materials)
        board = BoardBuilder.random_board(rng, army(material), [], black_king_squares=EDGE_SQUARES)
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle):
            return None
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        stalemating = [m for m in moves if self.oracle.draws_after(game, m) == STALEMATE]
        if not stalemating:
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=AvoidMoves.of(stalemating),
            difficulty=1 if len(stalemating) == 1 else 2,
            meta={"material": material, "stalemating": len(stalemating)},
        )


class RepetitionTrapGenerator(SamplingGenerator):
    """White is winning and has been shuffling a piece while Black shuffles its king.

    The shuffle is simulated from a Black-to-move position until White is to move and
    one of White's moves would repeat a position a third time. Starting the cycle on
    Black's move is what makes that possible: in a four-ply cycle the first position to
    come round three times is the cycle's first one, and here it is one White creates.
    """

    name = "repetition_trap"
    theme = "repetition_trap"
    description = "A won position after some shuffling: avoid the move that repeats it thrice."

    MAX_PLIES = 16

    def __init__(self, materials: tuple[str, ...] = ("Q", "QR", "RR"), **kwargs):
        super().__init__(**kwargs)
        self.materials = tuple(materials)
        for material in self.materials:
            army(material)
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        material = rng.choice(self.materials)
        board = BoardBuilder.random_board(rng, army(material), [])
        if board is None:
            return None
        start = board.position(Player.BLACK)
        game = start.to_game()
        if not is_quiet(game, self.oracle):
            return None

        king_move = self._shuffle(game, Player.BLACK, rng)
        if king_move is None:
            return None
        undo = game.make(*king_move)
        piece_move = self._shuffle(game, Player.WHITE, rng)
        game.unmake(undo)
        if piece_move is None:
            return None

        cycle = [king_move, piece_move, _back(king_move), _back(piece_move)]
        played: list[Move] = []
        for ply in range(self.MAX_PLIES):
            move = cycle[ply % 4]
            if game.turn == Player.WHITE and self.oracle.draws_after(game, move) == REPETITION:
                break
            if not self._keeps_calm(game, move):
                return None
            game.make(*move)
            played.append(move)
        else:
            return None

        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        repeating = [m for m in moves if self.oracle.draws_after(game, m) == REPETITION]
        if not repeating or len(repeating) == len(moves):
            return None
        position = Position(start.fen, tuple(move_to_uci(m) for m in played))
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=AvoidMoves.of(repeating),
            difficulty=1,
            meta={"material": material, "setup_plies": len(played)},
        )

    def _shuffle(self, game: ChessGame, player: Player, rng: random.Random) -> Move | None:
        """A quiet, reversible move for ``player``'s king (Black) or heavy piece (White)."""
        wanted = PieceType.KING if player == Player.BLACK else None
        candidates = []
        for source, dest in all_moves(game.grid, game.turn, game.en_passant_target):
            piece = game.grid[source[0]][source[1]]
            if piece is None or game.grid[dest[0]][dest[1]] is not None:
                continue
            if wanted is not None and piece.type != wanted:
                continue
            if wanted is None and piece.type in (PieceType.KING, PieceType.PAWN):
                continue
            candidates.append((source, dest))
        rng.shuffle(candidates)
        return next((m for m in candidates if self._keeps_calm(game, m)), None)

    def _keeps_calm(self, game: ChessGame, move: Move) -> bool:
        """Legal here, and afterwards neither king is en prise and the game goes on."""
        if not game.is_move_valid(*move):
            return False
        undo = game.make(*move)
        try:
            return is_quiet(game, self.oracle)
        finally:
            game.unmake(undo)


def _back(move: Move) -> Move:
    return (move[1], move[0])


__all__ = ["RepetitionTrapGenerator", "StalemateTrapGenerator", "army"]
