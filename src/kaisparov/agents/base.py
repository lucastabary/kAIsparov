"""Common policy interface shared by every agent.

A policy maps a game state to a move. Keeping this contract tiny lets the arena
pit any agent against any other — random, heuristic or neural — and lets the
trainer swap opponents without caring how a move is chosen.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kaisparov.core.board import ChessGame
from kaisparov.core.coords import Coord

Move = tuple[Coord, Coord]


@runtime_checkable
class Policy(Protocol):
    name: str

    def select_move(self, game: ChessGame) -> Move | None:
        """Return a legal ``(source, dest)`` move, or ``None`` if none is available."""
        ...
