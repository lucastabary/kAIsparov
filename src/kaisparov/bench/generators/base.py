"""Problem generators: the base classes every test family builds on.

:class:`ProblemGenerator` is the whole contract — "give me ``count`` problems from this
random stream". Subclasses register under their ``name`` on definition, so a suite
file can ask for ``generator: mate_in_one`` and nothing else has to know the class.

Most generators are *propose-and-verify* loops — sample a position, ask the
:class:`~kaisparov.bench.oracle.Oracle` whether it is a problem, keep it or try
again — and :class:`SamplingGenerator` is that loop, written once: it deduplicates,
balances colours by mirroring, gives up loudly when a generator's acceptance rate is
too low, and numbers the problems. A new family then only implements
:meth:`SamplingGenerator.propose`.

:class:`BoardBuilder` is the scratch board those proposals are drawn on.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any, ClassVar

from kaisparov.bench.oracle import Oracle
from kaisparov.bench.position import Position, other, to_fen
from kaisparov.bench.problem import Problem
from kaisparov.core.coords import ALL_SQUARES, Coord
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player
from kaisparov.core.rules import is_in_check

Grid = list[list["Piece | None"]]


class GenerationError(RuntimeError):
    """A generator could not produce the problems it was asked for."""


class ProblemGenerator(ABC):
    """Produces problems of one theme. Subclasses set ``name``/``theme``/``description``.

    Constructor keyword arguments are the generator's parameters, exactly as a suite
    file spells them under ``params:``; unknown ones are a ``TypeError``, which is the
    error a typo in a suite file should raise.
    """

    name: ClassVar[str] = ""
    theme: ClassVar[str] = ""
    description: ClassVar[str] = ""
    _registry: ClassVar[dict[str, type[ProblemGenerator]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = cls.__dict__.get("name")
        if not name:
            return  # an abstract intermediate class
        if name in ProblemGenerator._registry:
            raise TypeError(f"generator {name!r} is already registered")
        ProblemGenerator._registry[name] = cls

    @abstractmethod
    def generate(self, count: int | None, rng: random.Random) -> list[Problem]:
        """``count`` problems (``None``: the generator's natural amount), drawn from ``rng``."""

    # ---------------------------------------------------------------- registry
    @staticmethod
    def create(name: str, params: dict[str, Any] | None = None) -> ProblemGenerator:
        import kaisparov.bench.generators  # noqa: F401 - registers the built-in families

        if name not in ProblemGenerator._registry:
            known = ", ".join(sorted(ProblemGenerator._registry))
            raise ValueError(f"unknown generator {name!r} (known: {known})")
        return ProblemGenerator._registry[name](**(params or {}))

    @staticmethod
    def available() -> dict[str, type[ProblemGenerator]]:
        import kaisparov.bench.generators  # noqa: F401

        return dict(sorted(ProblemGenerator._registry.items()))


class SamplingGenerator(ProblemGenerator):
    """Propose-and-verify: call :meth:`propose` until ``count`` distinct problems exist.

    ``mirror`` flips a random half of the problems to the other colour, so a suite
    tests both sides even when :meth:`propose` always sets up White to move.
    ``attempts_per_problem`` bounds the loop: if fewer than one proposal in that many
    is accepted, the generator's constraints are too tight and it says so rather than
    spinning forever.
    """

    default_count: ClassVar[int] = 20

    def __init__(self, *, mirror: bool = True, attempts_per_problem: int = 500):
        self.mirror = mirror
        self.attempts_per_problem = attempts_per_problem

    @abstractmethod
    def propose(self, rng: random.Random) -> Problem | None:
        """One candidate (its ``id`` is ignored), or ``None`` to reject this draw."""

    def generate(self, count: int | None, rng: random.Random) -> list[Problem]:
        count = self.default_count if count is None else count
        problems: list[Problem] = []
        seen: set[Position] = set()
        budget = max(1, count) * self.attempts_per_problem
        for _ in range(budget):
            if len(problems) >= count:
                break
            candidate = self.propose(rng)
            if candidate is None or candidate.position in seen:
                continue
            if not _is_legal_position(candidate.position):
                continue
            seen.add(candidate.position)
            if self.mirror and rng.random() < 0.5:
                candidate = candidate.mirrored()
            problems.append(
                replace(
                    candidate,
                    id=f"{self.name}-{len(problems):04d}",
                    theme=candidate.theme or self.theme,
                    generator=self.name,
                )
            )
        if len(problems) < count:
            raise GenerationError(
                f"{self.name}: found {len(problems)}/{count} problems in {budget} attempts; "
                "loosen its parameters or raise attempts_per_problem"
            )
        return problems


def _is_legal_position(position: Position) -> bool:
    """Reject a position the rules of chess cannot reach.

    Specifically: the side *not* to move being in check. python-chess happily
    generates the capture of a king standing in check, so such a position would
    hand every contestant a free "solution" that says nothing about its play. Random
    board builders produce them regularly, hence the check here rather than in each
    generator.
    """
    game = position.to_game()
    return not is_in_check(game, other(game.turn))


# ------------------------------------------------------------------------ boards


PIECE_TYPES: tuple[PieceType, ...] = (
    PieceType.QUEEN,
    PieceType.ROOK,
    PieceType.BISHOP,
    PieceType.KNIGHT,
    PieceType.PAWN,
)


class BoardBuilder:
    """A scratch board to place pieces on at random, then freeze into a position.

    Placement follows the rules a *reachable* position obeys: pawns never stand on
    either back rank, and the two kings are never adjacent. Everything else — whether
    a king starts attacked, how much material each side has — is the generator's
    call, since that is precisely what problems differ in.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.grid: Grid = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]

    def free_squares(self, rows: Iterable[int] | None = None) -> list[Coord]:
        allowed = set(range(BOARD_SIZE) if rows is None else rows)
        return [(c, r) for c, r in ALL_SQUARES if r in allowed and self.grid[c][r] is None]

    def place(
        self,
        player: Player,
        piece_type: PieceType,
        square: Coord | None = None,
        *,
        rows: Iterable[int] | None = None,
        squares: Iterable[Coord] | None = None,
    ) -> Coord | None:
        """Put a piece on ``square``, or on a random allowed free square. ``None`` if full.

        ``rows`` and ``squares`` narrow the random choice (a king on the edge, a piece
        near the action); a piece that cannot legally stand anywhere left returns ``None``.
        """
        if square is None:
            candidates = [sq for sq in self.free_squares(rows) if self._allowed(piece_type, sq)]
            if squares is not None:
                wanted = set(squares)
                candidates = [sq for sq in candidates if sq in wanted]
            if piece_type == PieceType.KING:
                candidates = [sq for sq in candidates if not self._next_to_king(sq, player)]
            if not candidates:
                return None
            square = self.rng.choice(candidates)
        elif self.grid[square[0]][square[1]] is not None:
            raise ValueError(f"square {square} is occupied")
        elif not self._allowed(piece_type, square):
            return None
        self.grid[square[0]][square[1]] = Piece(player, piece_type)
        return square

    def remove(self, square: Coord) -> None:
        self.grid[square[0]][square[1]] = None

    def place_all(
        self,
        player: Player,
        pieces: Sequence[PieceType],
        *,
        rows: Iterable[int] | None = None,
        squares: Iterable[Coord] | None = None,
    ) -> bool:
        """Place every piece in ``pieces`` for ``player``; ``False`` if one did not fit."""
        rows = None if rows is None else list(rows)
        squares = None if squares is None else list(squares)
        return all(
            self.place(player, piece_type, rows=rows, squares=squares) is not None
            for piece_type in pieces
        )

    @classmethod
    def random_board(
        cls,
        rng: random.Random,
        white: Sequence[PieceType],
        black: Sequence[PieceType],
        *,
        black_king_squares: Iterable[Coord] | None = None,
    ) -> BoardBuilder | None:
        """Both kings, then ``white`` and ``black`` anywhere; ``None`` if something did not fit."""
        board = cls(rng)
        if board.place(Player.BLACK, PieceType.KING, squares=black_king_squares) is None:
            return None
        board.place(Player.WHITE, PieceType.KING)
        if not (board.place_all(Player.WHITE, white) and board.place_all(Player.BLACK, black)):
            return None
        return board

    def random_material(
        self, count: int, types: Sequence[PieceType] = PIECE_TYPES
    ) -> list[PieceType]:
        return [self.rng.choice(types) for _ in range(count)]

    def king_square(self, player: Player) -> Coord | None:
        for col, row in ALL_SQUARES:
            piece = self.grid[col][row]
            if piece is not None and piece.player == player and piece.type == PieceType.KING:
                return (col, row)
        return None

    def in_check(self, player: Player) -> bool:
        return is_in_check(self.grid, player)

    def position(self, turn: Player = Player.WHITE) -> Position:
        return Position(to_fen(self.grid, turn))

    def game(self, turn: Player = Player.WHITE) -> ChessGame:
        return self.position(turn).to_game()

    # ----------------------------------------------------------------- rules
    @staticmethod
    def _allowed(piece_type: PieceType, square: Coord) -> bool:
        return piece_type != PieceType.PAWN or 0 < square[1] < BOARD_SIZE - 1

    def _next_to_king(self, square: Coord, player: Player) -> bool:
        enemy = self.king_square(other(player))
        return enemy is not None and max(abs(enemy[0] - square[0]), abs(enemy[1] - square[1])) <= 1


# ---------------------------------------------------------------------- helpers

EDGE_SQUARES: tuple[Coord, ...] = tuple(
    (c, r) for c, r in ALL_SQUARES if c in (0, BOARD_SIZE - 1) or r in (0, BOARD_SIZE - 1)
)

HEAVY_TYPES: tuple[PieceType, ...] = (
    PieceType.QUEEN,
    PieceType.ROOK,
    PieceType.ROOK,
    PieceType.BISHOP,
    PieceType.KNIGHT,
)


def near(square: Coord, radius: int) -> list[Coord]:
    """Every square within ``radius`` king steps of ``square`` (itself excluded)."""
    return [
        (c, r)
        for c, r in ALL_SQUARES
        if (c, r) != square and max(abs(c - square[0]), abs(r - square[1])) <= radius
    ]


def is_quiet(game: ChessGame, oracle: Oracle) -> bool:
    """Neither king is en prise and the game is not over: a position to *think* in."""
    return (
        not game.is_in_check(Player.WHITE)
        and not game.is_in_check(Player.BLACK)
        and not oracle.is_over(game)
    )


__all__ = [
    "EDGE_SQUARES",
    "HEAVY_TYPES",
    "PIECE_TYPES",
    "BoardBuilder",
    "is_quiet",
    "near",
    "GenerationError",
    "ProblemGenerator",
    "SamplingGenerator",
]
