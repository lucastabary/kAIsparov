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


KNIGHT_TARGETS: dict[Coord, list[Coord]] = _step_targets(KNIGHT_OFFSETS)
KING_TARGETS: dict[Coord, list[Coord]] = _step_targets(KING_OFFSETS)
ORTHO_RAYS: dict[Coord, list[list[Coord]]] = _rays(ORTHO_DIRS)
DIAG_RAYS: dict[Coord, list[list[Coord]]] = _rays(DIAG_DIRS)
