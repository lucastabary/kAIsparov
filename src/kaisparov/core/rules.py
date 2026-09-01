"""Board-level rules that are not tied to a single piece's movement."""

from __future__ import annotations

from kaisparov.core.coords import Coord, all_squares
from kaisparov.core.movegen import Grid, pseudo_legal_moves
from kaisparov.core.pieces import PieceType, Player


def find_king(grid: Grid, player: Player) -> Coord | None:
    for x, y in all_squares():
        piece = grid[x][y]
        if piece is not None and piece.type == PieceType.KING and piece.player == player:
            return (x, y)
    return None


def is_in_check(grid: Grid, player: Player) -> bool:
    """True if ``player``'s king is attacked by any enemy piece."""
    king_pos = find_king(grid, player)
    if king_pos is None:
        return False

    enemy = Player.BLACK if player == Player.WHITE else Player.WHITE
    for x, y in all_squares():
        piece = grid[x][y]
        if (
            piece is not None
            and piece.player == enemy
            and king_pos in pseudo_legal_moves(grid, (x, y))
        ):
            return True
    return False
