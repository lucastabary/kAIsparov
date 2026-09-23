"""Curriculum utilities: generate simplified starting positions for training.

A curriculum controls how hard the positions the agent trains on are. Phase 0
ships a piece-count curriculum: random legal-ish positions with a bounded number
of pieces per side, optionally excluding major pieces (queen/rook).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player
from kaisparov.core.rules import is_in_check, is_legal_position

Grid = list[list[Piece | None]]


@dataclass(frozen=True)
class PhaseConfig:
    """Declarative description of one curriculum phase."""

    name: str
    max_pieces_per_side: int = 6  # total pieces per side, king included
    allow_major: bool = True  # queen / rook
    allow_minor: bool = True  # bishop / knight (KNIGHT)
    allow_pawns: bool = True
    # Place each king on a square that is NOT already attacked at ply 0, so the
    # random placement never starts a side in check. (A position with the side not
    # to move in check is redrawn regardless; see get_initial_board.) Kept on by default; turn off for the old raw-random behavior.
    ensure_kings_safe: bool = True
    # Pieces (king included) of the side the learner plays *against*. ``None`` gives a
    # balanced position (``max_pieces_per_side`` each, the learner's colour drawn at
    # random). Set it and the position is lopsided: a random colour gets the full
    # budget, the other only ``defender_pieces``, and the learner always plays the
    # strong side — ``defender_pieces: 1`` with majors only is a won endgame (KQ/KR vs K).
    defender_pieces: int | None = None

    def __post_init__(self) -> None:
        if self.defender_pieces is not None and not (
            1 <= self.defender_pieces <= self.max_pieces_per_side
        ):
            raise ValueError(
                f"curriculum phase {self.name!r}: defender_pieces must be between 1 (a "
                f"bare king) and max_pieces_per_side ({self.max_pieces_per_side}), "
                f"got {self.defender_pieces}"
            )

    def allowed_piece_types(self) -> list[PieceType]:
        types: list[PieceType] = []
        if self.allow_minor:
            types += [PieceType.BISHOP, PieceType.KNIGHT]
        if self.allow_major:
            types += [PieceType.ROOK, PieceType.QUEEN]
        if self.allow_pawns:
            types.append(PieceType.PAWN)
        if not types:
            # Always leave at least one non-king option so a position is playable.
            types.append(PieceType.PAWN)
        return types


class BaseCurriculum(ABC):
    """Interface for anything that can produce a starting board."""

    @abstractmethod
    def get_initial_board(self) -> Grid:
        """Return a fresh grid[col][row] usable as ChessGame(initial_board=...)."""

    def get_start(self) -> tuple[Grid, Player | None]:
        """A fresh grid, plus the side the learner must play (``None``: either).

        A lopsided position is only a lesson for its strong side; the rollout against
        an opponent seats the learner there instead of drawing its colour.
        """
        return self.get_initial_board(), None


class PieceCountCurriculum(BaseCurriculum):
    """Random positions with a bounded number of pieces per side.

    Both kings are always placed. Each side receives up to
    ``phase.max_pieces_per_side - 1`` additional pieces drawn from the phase's
    allowed types. White pieces live on the bottom half of the board, Black on
    the top half; pawns never spawn on the back ranks.
    """

    def __init__(self, phase: PhaseConfig, seed: int | None = None):
        self.phase = phase
        self._rng = random.Random(seed)

    # Draws before giving up on a legal position. With ensure_kings_safe on, the first
    # draw is legal essentially always; this only bounds the rare dense-board fallback
    # and the ensure_kings_safe=False case.
    MAX_DRAWS = 100

    def get_initial_board(self) -> Grid:
        return self.get_start()[0]

    def get_start(self) -> tuple[Grid, Player | None]:
        """A random position with White to move, never one chess cannot reach.

        Specifically, Black is never left in check: with White to move that position
        is illegal, and python-chess would happily generate the capture of the black
        king — handing White a "move" worth the king's sentinel material value and a
        game that then runs on without a king.

        With ``defender_pieces`` set, also returns the strong side (drawn at random,
        so the learner trains as both colours); otherwise ``None``.
        """
        strong: Player | None = None
        if self.phase.defender_pieces is not None:
            strong = self._rng.choice([Player.WHITE, Player.BLACK])
        for _ in range(self.MAX_DRAWS):
            grid = self._draw_board(strong)
            if is_legal_position(grid, Player.WHITE):
                return grid, strong
        raise RuntimeError(
            f"curriculum phase {self.phase.name!r}: no legal position in "
            f"{self.MAX_DRAWS} draws; lower max_pieces_per_side or enable ensure_kings_safe"
        )

    def _budget(self, player: Player, strong: Player | None) -> int:
        """Pieces (king included) ``player`` gets in this draw."""
        if strong is None or player == strong or self.phase.defender_pieces is None:
            return self.phase.max_pieces_per_side
        return self.phase.defender_pieces

    def _draw_board(self, strong: Player | None = None) -> Grid:
        grid: Grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        white_rows = range(0, BOARD_SIZE // 2)  # 0..3
        black_rows = range(BOARD_SIZE // 2, BOARD_SIZE)  # 4..7
        pawn_rows = range(1, BOARD_SIZE - 1)  # never rank 0 or 7

        # Non-king pieces first, then the kings — so a king can be placed on a square
        # that accounts for every other piece already on the board (needed to keep it
        # off an attacked square when ``ensure_kings_safe``). Black's king goes down
        # first and White's second: White's safe-square check then also sees Black's
        # king, which keeps the two kings from being placed adjacent (mutual capture).
        white_n = self._budget(Player.WHITE, strong)
        black_n = self._budget(Player.BLACK, strong)
        self._place_extras(grid, Player.WHITE, white_rows, pawn_rows, white_n)
        self._place_extras(grid, Player.BLACK, black_rows, pawn_rows, black_n)
        self._place_king(grid, Player.BLACK, black_rows)
        self._place_king(grid, Player.WHITE, white_rows)
        return grid

    def _place_extras(self, grid: Grid, player: Player, rows, pawn_rows, n_pieces: int) -> None:
        allowed = self.phase.allowed_piece_types()
        n_extra = max(0, n_pieces - 1)
        for _ in range(n_extra):
            piece_type = self._rng.choice(allowed)
            candidate_rows = pawn_rows if piece_type == PieceType.PAWN else rows
            self._place_piece(grid, player, piece_type, candidate_rows)

    def _place_king(self, grid: Grid, player: Player, rows) -> None:
        free = [(col, row) for col in range(BOARD_SIZE) for row in rows if grid[col][row] is None]
        if not free:
            return
        if self.phase.ensure_kings_safe:
            # Try free squares in random order; keep the first that leaves the king
            # unattacked. On a very dense board no such square may exist — fall through
            # to a plain random placement rather than fail to place a king.
            self._rng.shuffle(free)
            for col, row in free:
                grid[col][row] = Piece(player, PieceType.KING)
                if not is_in_check(grid, player):
                    return
                grid[col][row] = None
        col, row = self._rng.choice(free)
        grid[col][row] = Piece(player, PieceType.KING)

    def _place_piece(self, grid: Grid, player: Player, piece_type: PieceType, rows) -> None:
        free = [(col, row) for col in range(BOARD_SIZE) for row in rows if grid[col][row] is None]
        if not free:
            return
        col, row = self._rng.choice(free)
        grid[col][row] = Piece(player, piece_type)
