"""Board-level rules that are not tied to a single piece's movement."""

from __future__ import annotations

from kaisparov.core import attacks
from kaisparov.core.coords import ALL_SQUARES, Coord, in_bounds
from kaisparov.core.movegen import Grid
from kaisparov.core.pieces import PieceType, Player


def find_king(grid: Grid, player: Player) -> Coord | None:
    for x, y in ALL_SQUARES:
        piece = grid[x][y]
        if piece is not None and piece.type == PieceType.KING and piece.player == player:
            return (x, y)
    return None


def is_in_check(grid: Grid, player: Player) -> bool:
    """True if ``player``'s king is attacked by any enemy piece.

    Scans *outward from the king* along each attack pattern and returns on the first
    enemy attacker, reusing the precomputed :mod:`kaisparov.core.attacks` tables: a
    slider ray stops at its first blocker and no enemy move lists are ever built. This
    is the same blocking-aware notion as before (equivalent to asking whether any enemy
    ``pseudo_legal_moves`` reaches the king) at a fraction of the cost.
    """
    king_pos = find_king(grid, player)
    if king_pos is None:
        return False

    enemy = Player.BLACK if player == Player.WHITE else Player.WHITE
    kx, ky = king_pos

    # Knight: an enemy knight on any knight-hop square attacks the king.
    for tx, ty in attacks.KNIGHT_TARGETS[king_pos]:
        piece = grid[tx][ty]
        if piece is not None and piece.player == enemy and piece.type == PieceType.KNIGHT:
            return True

    # Adjacent enemy king (a king can capture an adjacent king in this variant).
    for tx, ty in attacks.KING_TARGETS[king_pos]:
        piece = grid[tx][ty]
        if piece is not None and piece.player == enemy and piece.type == PieceType.KING:
            return True

    # Enemy pawns attack diagonally toward the king: a black pawn sits one row above
    # the king (it captures downward), a white pawn one row below.
    pawn_dy = 1 if enemy == Player.BLACK else -1
    for dx in (-1, 1):
        target = (kx + dx, ky + pawn_dy)
        if in_bounds(target):
            piece = grid[target[0]][target[1]]
            if piece is not None and piece.player == enemy and piece.type == PieceType.PAWN:
                return True

    # Sliders: the first piece down each ray. Orthogonal -> enemy rook/queen;
    # diagonal -> enemy bishop/queen. A blocker of any kind ends the ray.
    for ray in attacks.ORTHO_RAYS[king_pos]:
        for cx, cy in ray:
            piece = grid[cx][cy]
            if piece is not None:
                if piece.player == enemy and piece.type in (PieceType.ROOK, PieceType.QUEEN):
                    return True
                break
    for ray in attacks.DIAG_RAYS[king_pos]:
        for cx, cy in ray:
            piece = grid[cx][cy]
            if piece is not None:
                if piece.player == enemy and piece.type in (PieceType.BISHOP, PieceType.QUEEN):
                    return True
                break

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
