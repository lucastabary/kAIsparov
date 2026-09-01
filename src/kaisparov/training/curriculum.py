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

Grid = list[list[Piece | None]]


@dataclass(frozen=True)
class PhaseConfig:
    """Declarative description of one curriculum phase."""

    name: str
    max_pieces_per_side: int = 6  # total pieces per side, king included
    allow_major: bool = True  # queen / rook
    allow_minor: bool = True  # bishop / knight (KNIGHT)
    allow_pawns: bool = True

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

    def get_initial_board(self) -> Grid:
        grid: Grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

        white_rows = range(0, BOARD_SIZE // 2)  # 0..3
        black_rows = range(BOARD_SIZE // 2, BOARD_SIZE)  # 4..7
        pawn_rows = range(1, BOARD_SIZE - 1)  # never rank 0 or 7

        self._place_side(grid, Player.WHITE, white_rows, pawn_rows)
        self._place_side(grid, Player.BLACK, black_rows, pawn_rows)
        return grid

    def _place_side(self, grid: Grid, player: Player, rows, pawn_rows) -> None:
        allowed = self.phase.allowed_piece_types()
        n_extra = max(0, self.phase.max_pieces_per_side - 1)

        self._place_piece(grid, player, PieceType.KING, rows)
        for _ in range(n_extra):
            piece_type = self._rng.choice(allowed)
            candidate_rows = pawn_rows if piece_type == PieceType.PAWN else rows
            self._place_piece(grid, player, piece_type, candidate_rows)

    def _place_piece(self, grid: Grid, player: Player, piece_type: PieceType, rows) -> None:
        free = [(col, row) for col in range(BOARD_SIZE) for row in rows if grid[col][row] is None]
        if not free:
            return
        col, row = self._rng.choice(free)
        grid[col][row] = Piece(player, piece_type)
