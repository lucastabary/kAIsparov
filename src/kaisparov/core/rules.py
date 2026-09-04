"""Board-level rules that are not tied to a single piece's movement."""

from __future__ import annotations

from kaisparov.core import attacks
from kaisparov.core.coords import ALL_SQUARES, Coord, in_bounds
from kaisparov.core.movegen import Grid, pseudo_legal_moves
from kaisparov.core.pieces import PieceType, Player


def find_king(grid: Grid, player: Player) -> Coord | None:
    for x, y in ALL_SQUARES:
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
    for x, y in ALL_SQUARES:
        piece = grid[x][y]
        if (
            piece is not None
            and piece.player == enemy
            and king_pos in pseudo_legal_moves(grid, (x, y))
        ):
            return True
    return False


def _slider_rays(piece_type: PieceType, source: Coord) -> list[list[Coord]]:
    if piece_type == PieceType.ROOK:
        return attacks.ORTHO_RAYS[source]
    if piece_type == PieceType.BISHOP:
        return attacks.DIAG_RAYS[source]
    return attacks.QUEEN_RAYS[source]  # QUEEN


def attacked_squares(grid: Grid, by_player: Player) -> set[Coord]:
    """Return every square ``by_player`` controls in this position.

    A square is controlled if one of ``by_player``'s pieces could capture a piece
    standing there. Sliding pieces stop at the first piece on each ray (that
    blocker's square is attacked; squares behind it are not), so this is the same
    blocking-aware notion :func:`is_in_check` uses — but exposed for *every* square,
    empty ones included, so a caller can also test whether a would-be destination
    (e.g. a king's escape square) is safe. Pawns control only their two forward
    diagonals, never the push square; the king/knight control their step targets.

    Unlike :func:`kaisparov.core.movegen.pseudo_legal_moves`, occupancy of the
    target square is irrelevant here: a square an enemy pawn guards diagonally is
    attacked whether it is empty, friendly, or hostile.
    """
    controlled: set[Coord] = set()
    for x, y in ALL_SQUARES:
        piece = grid[x][y]
        if piece is None or piece.player != by_player:
            continue

        if piece.type == PieceType.KNIGHT:
            controlled.update(attacks.KNIGHT_TARGETS[(x, y)])
        elif piece.type == PieceType.KING:
            controlled.update(attacks.KING_TARGETS[(x, y)])
        elif piece.type == PieceType.PAWN:
            direction = 1 if piece.player == Player.WHITE else -1
            for dx in (-1, 1):
                target = (x + dx, y + direction)
                if in_bounds(target):
                    controlled.add(target)
        else:  # QUEEN / ROOK / BISHOP
            for ray in _slider_rays(piece.type, (x, y)):
                for cx, cy in ray:
                    controlled.add((cx, cy))
                    if grid[cx][cy] is not None:
                        break  # blocked: attacks the blocker, nothing beyond it
    return controlled
