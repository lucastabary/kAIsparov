"""Tactics: positions with a concrete, forcing answer the oracle can prove.

- ``win_in_n`` — a king capture forced in exactly ``depth`` moves;
- ``free_capture`` — an enemy piece can be taken for free;
- ``fork`` — a quiet move attacks two targets and wins material by force;
- ``parry_threat`` — the opponent threatens a forced win; find a move that stops it.

Every generator here proposes White to move and lets the base class mirror half the
problems, and every answer is either enumerated by an exhaustive search or (for the
material themes) graded by searching the move the contestant actually plays.
"""

from __future__ import annotations

import random

from kaisparov.bench.generators.base import (
    EDGE_SQUARES,
    HEAVY_TYPES,
    PIECE_TYPES,
    BoardBuilder,
    SamplingGenerator,
    is_quiet,
)
from kaisparov.bench.oracle import WIN, Oracle
from kaisparov.bench.position import move_to_uci
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import FindMove, WinMaterial
from kaisparov.core.game import ChessGame
from kaisparov.core.move import Move
from kaisparov.core.pieces import PieceType, Player
from kaisparov.core.rules import pawn_attacks


def _captures(game: ChessGame) -> list[Move]:
    moves = game.legal_moves()
    return [m for m in moves if game.grid[m[1][0]][m[1][1]] is not None]


class WinInNGenerator(SamplingGenerator):
    name = "win_in_n"
    theme = "win_in_n"
    description = "Force a king capture in exactly N moves (N=2 is a classical mate in one)."

    def __init__(
        self,
        depth: int = 2,
        min_pieces: int = 2,
        max_pieces: int = 5,
        max_solutions: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if depth < 2:
            raise ValueError("win_in_n needs depth >= 2 (depth 1 is mate_in_one)")
        self.depth = depth
        self.min_pieces, self.max_pieces = min_pieces, max_pieces
        self.max_solutions = max_solutions
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        white = [
            rng.choice(HEAVY_TYPES) for _ in range(rng.randint(self.min_pieces, self.max_pieces))
        ]
        black = [rng.choice(PIECE_TYPES) for _ in range(rng.randint(0, self.max_pieces - 1))]
        # A king on the edge has fewer flight squares, which is where mates live.
        edge = EDGE_SQUARES if rng.random() < 0.7 else None
        board = BoardBuilder.random_board(rng, white, black, black_king_squares=edge)
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle) or self.oracle.win_depth(game, self.depth) != self.depth:
            return None
        winners = self.oracle.winning_moves(game, self.depth)
        if not winners or len(winners) > self.max_solutions:
            return None
        return Problem(
            id="",
            theme=f"win_in_{self.depth}",
            position=position,
            task=FindMove.of(winners),
            difficulty=self.depth,
            meta={"solutions": len(winners)},
        )


class FreeCaptureGenerator(SamplingGenerator):
    name = "free_capture"
    theme = "free_capture"
    description = "An enemy piece hangs: win it (graded by searching the move played)."

    def __init__(self, min_gain: float = 3.0, plies: int = 3, max_pieces: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.min_gain, self.plies, self.max_pieces = min_gain, plies, max_pieces
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        white = board_material(rng, 2, self.max_pieces)
        black = board_material(rng, 2, self.max_pieces)
        board = BoardBuilder.random_board(rng, white, black)
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle):
            return None
        gains = {m: self.oracle.material_gain(game, m, self.plies) for m in _captures(game)}
        winning = [m for m, gain in gains.items() if self.min_gain <= gain < WIN]
        if not winning:
            return None
        best = max(gains[m] for m in winning)
        if any(self.oracle.draws_after(game, m) for m in winning):
            return None  # taking it would end the game drawn: not the lesson here
        poisoned = sum(1 for gain in gains.values() if gain < 0)
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=WinMaterial(
                self.min_gain, best, self.plies, tuple(move_to_uci(m) for m in winning)
            ),
            difficulty=1 + min(2, poisoned),  # captures that lose material are the trap
            meta={"best_gain": best, "poisoned_captures": poisoned},
        )


class ForkGenerator(SamplingGenerator):
    name = "fork"
    theme = "fork"
    description = "A quiet move attacks two targets and wins material by force."

    ATTACKERS = (
        PieceType.KNIGHT,
        PieceType.KNIGHT,
        PieceType.QUEEN,
        PieceType.BISHOP,
        PieceType.ROOK,
        PieceType.PAWN,
    )

    def __init__(self, min_gain: float = 2.0, plies: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.min_gain, self.plies = min_gain, plies
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        board = BoardBuilder(rng)
        attacker = rng.choice(self.ATTACKERS)
        fork_square = board.place(Player.WHITE, attacker)
        if fork_square is None:
            return None
        if attacker == PieceType.PAWN:
            targets = list(
                pawn_attacks(fork_square, Player.WHITE)
            )  # on an empty board, no captures show
        else:
            targets = board.game().possible_moves(fork_square)
        if len(targets) < 2:
            return None
        board.remove(fork_square)

        # Two targets: the king and a heavy piece, or two heavy pieces.
        first, second = rng.sample(targets, 2)
        if rng.random() < 0.5:
            board.place(Player.BLACK, PieceType.KING, first)
        else:
            board.place(Player.BLACK, rng.choice((PieceType.QUEEN, PieceType.ROOK)), first)
        if board.place(Player.BLACK, rng.choice((PieceType.QUEEN, PieceType.ROOK)), second) is None:
            return None
        if (
            board.king_square(Player.BLACK) is None
            and board.place(Player.BLACK, PieceType.KING) is None
        ):
            return None

        # The forking piece starts on a square from which it can reach the fork square.
        origins = []
        for square in board.free_squares():
            if board.place(Player.WHITE, attacker, square) is None:
                continue
            if fork_square in board.game().possible_moves(square) and not board.in_check(
                Player.BLACK
            ):
                origins.append(square)
            board.remove(square)
        if not origins:
            return None
        origin = rng.choice(origins)
        board.place(Player.WHITE, attacker, origin)
        if board.place(Player.WHITE, PieceType.KING) is None:
            return None
        board.place_all(Player.WHITE, [rng.choice(PIECE_TYPES) for _ in range(rng.randint(0, 3))])
        board.place_all(Player.BLACK, [rng.choice(PIECE_TYPES) for _ in range(rng.randint(0, 3))])

        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle):
            return None
        fork = Move(origin, fork_square)
        gain = self.oracle.material_gain(game, fork, self.plies)
        if not self.min_gain <= gain < WIN:
            return None
        # The fork must be the lesson: no capture already wins as much on the spot.
        if any(
            self.oracle.material_gain(game, m, self.plies) >= self.min_gain for m in _captures(game)
        ):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=WinMaterial(self.min_gain, gain, self.plies, (move_to_uci(fork),)),
            difficulty=1 if attacker == PieceType.KNIGHT else 2,
            meta={"attacker": attacker.name.lower(), "gain": gain},
        )


class ParryThreatGenerator(SamplingGenerator):
    name = "parry_threat"
    theme = "parry_threat"
    description = "The opponent threatens a forced king capture: find a move that stops it."

    def __init__(self, depth: int = 2, max_fraction: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.depth, self.max_fraction = depth, max_fraction
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        white = [rng.choice(PIECE_TYPES) for _ in range(rng.randint(1, 4))]
        black = [rng.choice(HEAVY_TYPES) for _ in range(rng.randint(2, 4))]
        board = BoardBuilder(rng)
        board.place(Player.WHITE, PieceType.KING, squares=EDGE_SQUARES)
        board.place(Player.BLACK, PieceType.KING)
        if not (board.place_all(Player.WHITE, white) and board.place_all(Player.BLACK, black)):
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle) or not self.oracle.threatens(game, self.depth):
            return None
        if self.oracle.wins_within(game, self.depth):
            return None  # White's own attack comes first: a different lesson

        legal = game.legal_moves()
        parries = []
        for move in self.oracle.safe_moves(game):
            undo = game.make(*move)
            try:
                if not self.oracle.wins_within(game, self.depth):
                    parries.append(move)
            finally:
                game.unmake(undo)
        if not parries or len(parries) > max(1, self.max_fraction * len(legal)):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=FindMove.of(parries),
            difficulty=1 if len(parries) > 2 else 2 if len(parries) == 2 else 3,
            meta={"parries": len(parries), "legal": len(legal)},
        )


def board_material(rng: random.Random, low: int, high: int) -> list[PieceType]:
    """A plausible army: ``low..high`` pieces, pawns about as common as everything else."""
    weighted = (*PIECE_TYPES, PieceType.PAWN, PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP)
    return [rng.choice(weighted) for _ in range(rng.randint(low, high))]


__all__ = [
    "ForkGenerator",
    "FreeCaptureGenerator",
    "ParryThreatGenerator",
    "WinInNGenerator",
    "board_material",
]
