"""Precomputed movement geometry.

These tables are built once at import time and turn move generation into simple
lookups instead of per-call range construction with bounds checks. Everything is
keyed by a ``(col, row)`` square.

* ``KNIGHT_TARGETS`` / ``KING_TARGETS`` — the on-board squares one hop away.
* ``ORTHO_RAYS`` / ``DIAG_RAYS`` — for each square, a list of rays; each ray is a
  list of squares ordered from nearest to farthest. Sliding pieces walk a ray and
  stop at the first occupied square.
"""

from __future__ import annotations

from kaisparov.core.coords import Coord, all_squares, in_bounds

KNIGHT_OFFSETS = [(2, -1), (2, 1), (1, -2), (1, 2), (-2, 1), (-2, -1), (-1, -2), (-1, 2)]
KING_OFFSETS = [(0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)]
ORTHO_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG_DIRS = [(1, 1), (-1, -1), (1, -1), (-1, 1)]


def _step_targets(offsets: list[tuple[int, int]]) -> dict[Coord, list[Coord]]:
    table: dict[Coord, list[Coord]] = {}
    for x, y in all_squares():
        table[(x, y)] = [(x + dx, y + dy) for dx, dy in offsets if in_bounds((x + dx, y + dy))]
    return table


def _rays(directions: list[tuple[int, int]]) -> dict[Coord, list[list[Coord]]]:
    table: dict[Coord, list[list[Coord]]] = {}
    for x, y in all_squares():
        rays: list[list[Coord]] = []
        for dx, dy in directions:
            ray: list[Coord] = []
            cx, cy = x + dx, y + dy
            while in_bounds((cx, cy)):
                ray.append((cx, cy))
                cx += dx
                cy += dy
            if ray:
                rays.append(ray)
        table[(x, y)] = rays
    return table


def _pawn_pushes(direction: int) -> dict[Coord, tuple[Coord | None, Coord | None]]:
    """Per-square (single, double) forward push destinations, ``None`` when off-board.

    Pure geometry; the caller still checks occupancy and ``has_moved``.
    """
    table: dict[Coord, tuple[Coord | None, Coord | None]] = {}
    for x, y in all_squares():
        one = (x, y + direction)
        two = (x, y + 2 * direction)
        table[(x, y)] = (
            one if in_bounds(one) else None,
            two if in_bounds(two) else None,
        )
    return table


def _pawn_captures(direction: int) -> dict[Coord, list[Coord]]:
    """Per-square forward diagonal squares (on-board only) a pawn can capture onto."""
    table: dict[Coord, list[Coord]] = {}
    for x, y in all_squares():
        caps = [(x + dx, y + direction) for dx in (-1, 1) if in_bounds((x + dx, y + direction))]
        table[(x, y)] = caps
    return table


KNIGHT_TARGETS: dict[Coord, list[Coord]] = _step_targets(KNIGHT_OFFSETS)
KING_TARGETS: dict[Coord, list[Coord]] = _step_targets(KING_OFFSETS)
ORTHO_RAYS: dict[Coord, list[list[Coord]]] = _rays(ORTHO_DIRS)
DIAG_RAYS: dict[Coord, list[list[Coord]]] = _rays(DIAG_DIRS)
# Queen rays = orthogonals + diagonals, concatenated once instead of on every call.
QUEEN_RAYS: dict[Coord, list[list[Coord]]] = {
    sq: ORTHO_RAYS[sq] + DIAG_RAYS[sq] for sq in ORTHO_RAYS
}
# Pawn geometry, mirroring the knight/king lookup tables. Split per colour into
# separate constants (rather than a dict keyed by the Player enum) so the hot path
# selects with a cheap identity branch instead of hashing an Enum member on every call.
PAWN_PUSHES_WHITE: dict[Coord, tuple[Coord | None, Coord | None]] = _pawn_pushes(1)
PAWN_PUSHES_BLACK: dict[Coord, tuple[Coord | None, Coord | None]] = _pawn_pushes(-1)
PAWN_CAPTURES_WHITE: dict[Coord, list[Coord]] = _pawn_captures(1)
PAWN_CAPTURES_BLACK: dict[Coord, list[Coord]] = _pawn_captures(-1)
