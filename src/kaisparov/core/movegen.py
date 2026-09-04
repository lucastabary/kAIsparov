"""Pseudo-legal move generation.

Pure functions over a ``grid`` (``list[list[Piece | None]]``). "Pseudo-legal"
means moves are geometrically valid and respect captures/blocking, but — matching
this project's capture-the-king variant — moves are *not* filtered for leaving
one's own king in check.

En passant needs one bit of state beyond the grid (the square a pawn just skipped),
so it is passed in as ``en_passant_target``. Castling is only offered from the
standard king start square (both king and rook unmoved and on their home squares).
"""

from __future__ import annotations

from kaisparov.core import attacks
from kaisparov.core.coords import Coord
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player

Grid = list[list["Piece | None"]]
Move = tuple[Coord, Coord]

KING_START_COL = BOARD_SIZE // 2  # e-file (index 4)


def _sliding_rays(source: Coord, piece_type: PieceType) -> list[list[Coord]]:
    if piece_type == PieceType.ROOK:
        return attacks.ORTHO_RAYS[source]
    if piece_type == PieceType.BISHOP:
        return attacks.DIAG_RAYS[source]
    return attacks.QUEEN_RAYS[source]  # QUEEN


def _castling_moves(grid: Grid, source: Coord, player: Player) -> list[Coord]:
    x, y = source
    home_rank = 0 if player == Player.WHITE else BOARD_SIZE - 1
    # Only well-defined from the standard king square; guards random curriculum boards.
    if x != KING_START_COL or y != home_rank:
        return []

    moves: list[Coord] = []
    for rook_x, dest_x, inner in (
        (BOARD_SIZE - 1, x + 2, range(x + 1, BOARD_SIZE - 1)),  # kingside
        (0, x - 2, range(1, x)),  # queenside
    ):
        rook = grid[rook_x][y]
        if rook is None or rook.type != PieceType.ROOK or rook.player != player or rook.has_moved:
            continue
        if all(grid[bx][y] is None for bx in inner):
            moves.append((dest_x, y))
    return moves


def pseudo_legal_moves(
    grid: Grid, source: Coord, en_passant_target: Coord | None = None
) -> list[Coord]:
    """Return the destination squares reachable by the piece on ``source``."""
    x, y = source
    piece = grid[x][y]
    if piece is None:
        return []

    player = piece.player
    moves: list[Coord] = []

    if piece.type == PieceType.KNIGHT:
        for dx, dy in attacks.KNIGHT_TARGETS[source]:
            target = grid[dx][dy]
            if target is None or target.player != player:
                moves.append((dx, dy))
        return moves

    if piece.type == PieceType.KING:
        for dx, dy in attacks.KING_TARGETS[source]:
            target = grid[dx][dy]
            if target is None or target.player != player:
                moves.append((dx, dy))
        if not piece.has_moved:
            moves.extend(_castling_moves(grid, source, player))
        return moves

    if piece.type in (PieceType.QUEEN, PieceType.BISHOP, PieceType.ROOK):
        for ray in _sliding_rays(source, piece.type):
            for cx, cy in ray:
                target = grid[cx][cy]
                if target is None:
                    moves.append((cx, cy))
                    continue
                if target.player != player:
                    moves.append((cx, cy))
                break  # stop at the first piece on the ray
        return moves

    if piece.type == PieceType.PAWN:
        if player == Player.WHITE:
            pushes, captures = attacks.PAWN_PUSHES_WHITE, attacks.PAWN_CAPTURES_WHITE
        else:
            pushes, captures = attacks.PAWN_PUSHES_BLACK, attacks.PAWN_CAPTURES_BLACK

        one, two = pushes[source]
        if one is not None and grid[one[0]][one[1]] is None:
            moves.append(one)
            if not piece.has_moved and two is not None and grid[two[0]][two[1]] is None:
                moves.append(two)

        for capture in captures[source]:
            target = grid[capture[0]][capture[1]]
            if target is not None and target.player != player:
                moves.append(capture)  # normal diagonal capture
            elif target is None and capture == en_passant_target:
                moves.append(capture)  # en passant capture (onto the empty skipped square)
        return moves

    return moves


def all_moves(grid: Grid, player: Player, en_passant_target: Coord | None = None) -> list[Move]:
    """Every pseudo-legal ``(source, dest)`` move available to ``player``."""
    moves: list[Move] = []
    for x in range(BOARD_SIZE):
        col = grid[x]  # hoist the column lookup out of the inner loop
        for y in range(BOARD_SIZE):
            piece = col[y]
            if piece is None or piece.player != player:
                continue
            square = (x, y)
            for dest in pseudo_legal_moves(grid, square, en_passant_target):
                moves.append((square, dest))
    return moves
