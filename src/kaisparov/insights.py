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
from enum import Enum
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


# ------------------------------------------------------------------ move review


class MoveQuality(Enum):
    """How good a played move was, in the chess.com "game review" vocabulary.

    Purely a *label*: the numbers it is derived from (how much winning chance the
    move gave away, whether it sacrificed material) live on :class:`MoveVerdict`.
    Members are declared best-to-worst, which is the order the end-of-game summary
    lists them in.
    """

    BRILLIANT = "brilliant"
    GREAT = "great"
    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    FORCED = "forced"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    MISS = "miss"
    BLUNDER = "blunder"

    @property
    def symbol(self) -> str:
        """Two characters at most — what the badge on the board shows."""
        return _QUALITY_SYMBOLS[self]

    @property
    def caption(self) -> str:
        """Name for the side panel (French, ASCII-only like the rest of the UI)."""
        return _QUALITY_CAPTIONS[self]


_QUALITY_SYMBOLS: dict[MoveQuality, str] = {
    MoveQuality.BRILLIANT: "!!",
    MoveQuality.GREAT: "!",
    MoveQuality.BEST: "*",
    MoveQuality.EXCELLENT: "+",
    MoveQuality.GOOD: "=",
    MoveQuality.FORCED: "F",
    MoveQuality.INACCURACY: "?!",
    MoveQuality.MISTAKE: "?",
    MoveQuality.MISS: "X",
    MoveQuality.BLUNDER: "??",
}

_QUALITY_CAPTIONS: dict[MoveQuality, str] = {
    MoveQuality.BRILLIANT: "Brillant",
    MoveQuality.GREAT: "Tres bon",
    MoveQuality.BEST: "Meilleur coup",
    MoveQuality.EXCELLENT: "Excellent",
    MoveQuality.GOOD: "Bon",
    MoveQuality.FORCED: "Force",
    MoveQuality.INACCURACY: "Imprecision",
    MoveQuality.MISTAKE: "Erreur",
    MoveQuality.MISS: "Occasion manquee",
    MoveQuality.BLUNDER: "Gaffe",
}


@dataclass(frozen=True, slots=True)
class MoveVerdict:
    """The review of one played move, from the mover's point of view.

    ``win_prob_before`` is the mover's winning chance had they played the top move;
    ``win_prob_after`` is their winning chance after the move actually played. The
    inputs the label was derived from are kept so the UI can show the reasoning
    rather than only the verdict.
    """

    move: Move
    quality: MoveQuality
    win_prob_before: float  # 0..1
    win_prob_after: float  # 0..1
    best_move: Move | None = None
    sacrificed: float = 0.0  # material (in pawns) the move hands to the opponent
    source: str = ""  # which evaluator judged it

    @property
    def loss(self) -> float:
        """Winning chance given away, in percentage points (never negative)."""
        return max(0.0, 100.0 * (self.win_prob_before - self.win_prob_after))


@runtime_checkable
class Judge(Protocol):
    """Anything that can grade a move before it is played. See :mod:`kaisparov.analysis`."""

    name: str

    def judge(self, game, move: Move) -> MoveVerdict | None:
        """Grade ``move`` in ``game`` — the position *before* the move is played."""
        ...
