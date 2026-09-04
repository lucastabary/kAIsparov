"""Vocabulary for "why did the model pick that?" — the data a developer mode
surfaces while a game is played.

Kept deliberately torch-free so the pygame UI can *render* an analysis without
pulling in a model backend, while the agents that *produce* one stay on the torch
side. An :class:`Analyzer` turns a position into a :class:`PositionAnalysis`: the
critic's scalar read of the side to move plus a ranked list of candidate moves.
Neither the renderer nor the producer imports the other — they only share these
plain dataclasses.

This is the seam a future backend hooks into: expose principal-variation lines,
attention weights, or search trees by producing richer :class:`PositionAnalysis`
values; the UI overlay grows to match without the engine or agents changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kaisparov.core.coords import Coord

Move = tuple[Coord, Coord]


@dataclass(frozen=True, slots=True)
class MoveInsight:
    """A single candidate move the model finds worth attention."""

    move: Move
    score: float  # relative weight in ``0..1`` (best candidate = 1.0), for overlay intensity
    label: str = ""  # short human caption, e.g. a probability like ``"62%"``


@dataclass(frozen=True, slots=True)
class PositionAnalysis:
    """What an analyzer thinks about the position for the side to move.

    ``candidates`` is ordered best-first. ``value`` is the critic's evaluation from
    the mover's point of view (positive = good for them), or ``None`` if the
    analyzer has no value head.
    """

    candidates: tuple[MoveInsight, ...] = ()
    value: float | None = None
    source: str = ""  # which analyzer produced this, for the panel/logs

    @property
    def best(self) -> MoveInsight | None:
        return self.candidates[0] if self.candidates else None


@runtime_checkable
class Analyzer(Protocol):
    """Anything that can explain a position. Mirrors :class:`~kaisparov.agents.base.Policy`."""

    name: str

    def analyze(self, game) -> PositionAnalysis | None:
        """Return an analysis for the side to move, or ``None`` if unavailable."""
        ...
