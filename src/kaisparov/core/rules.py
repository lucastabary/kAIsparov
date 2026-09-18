"""Board-level rules that are not tied to a single piece's movement.

Thin wrappers over python-chess, kept as free functions because two kinds of caller
need them: the ones holding a live :class:`~kaisparov.core.game.ChessGame` (analysis,
the models) and the ones holding only a hand-built ``grid`` that is not a game yet
(the curriculum sampler, the benchmark's board builder). Both are accepted; passing a
grid pays for building a throwaway :class:`chess.Board`, so hot paths should pass the
game — or, better, read its bitboards directly.
"""

from __future__ import annotations

import chess

from kaisparov.core.coords import ALL_SQUARES, Coord
from kaisparov.core.move import coord_to_square, square_to_coord
from kaisparov.core.pieces import Piece, PieceType, Player

Grid = list[list["Piece | None"]]
Position = "ChessGame | Grid"


def _as_board(position) -> chess.Board:
    """Accept a :class:`ChessGame` or a bare grid, return a python-chess board."""
    board = getattr(position, "board", None)
    if board is not None:
        return board
    from kaisparov.core.game import _board_from_grid

    return _board_from_grid(position, Player.WHITE, None)


def find_king(grid: Grid, player: Player) -> Coord | None:
    """Locate ``player``'s king on a bare grid, or ``None`` if it has none."""
    for x, y in ALL_SQUARES:
        piece = grid[x][y]
        if piece is not None and piece.type == PieceType.KING and piece.player == player:
            return (x, y)
    return None


def is_in_check(position, player: Player) -> bool:
    """True if ``player``'s king is attacked. A side without a king is never in check."""
    board = _as_board(position)
    king = board.king(player == Player.WHITE)
    if king is None:
        return False
    return board.is_attacked_by(player != Player.WHITE, king)


def attacked_squares(position, by_player: Player) -> set[Coord]:
    """Every square ``by_player`` controls in this position.

    A square is controlled if one of ``by_player``'s pieces could capture a piece
    standing there. Sliders stop at the first blocker (whose square *is* attacked);
    pawns control only their two forward diagonals, never the push square. Occupancy of
    the target is irrelevant, so this also answers "is that escape square safe?".
    """
    board = _as_board(position)
    colour = by_player == Player.WHITE
    controlled: set[Coord] = set()
    for square in chess.scan_forward(board.occupied_co[colour]):
        for target in board.attacks(square):
            controlled.add(square_to_coord(target))
    return controlled


def pawn_attacks(square: Coord, player: Player) -> tuple[Coord, ...]:
    """The squares a ``player`` pawn on ``square`` would capture on.

    Geometry only: it ignores what actually stands there, which is what a generator
    placing pieces on an empty board needs.
    """
    mask = chess.BB_PAWN_ATTACKS[player == Player.WHITE][coord_to_square(square)]
    return tuple(square_to_coord(s) for s in chess.scan_forward(mask))


__all__ = ["find_king", "is_in_check", "attacked_squares", "pawn_attacks"]
