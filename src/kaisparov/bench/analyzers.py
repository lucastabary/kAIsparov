"""Analyzers for the torch-free baselines, so probe tasks have a reference to beat.

A neural contestant analyses a position with its own heads
(:class:`~kaisparov.agents.neural_analyzer.NeuralAnalyzer`). The material baseline has
no heads, but it has an obvious "opinion": moves ranked by the material they win, and a
value that is the material balance. Probing it tells what a trained model's value and
policy add *beyond* counting pieces — the interesting part.
"""

from __future__ import annotations

import math

from kaisparov.analysis.evaluators import HeuristicEvaluator
from kaisparov.bench.oracle import Oracle
from kaisparov.core.game import ChessGame
from kaisparov.insights import MoveInsight, PositionAnalysis

_CLIP = 20.0  # pawns: a king capture (±WIN) weighs like a huge gain, not an overflow


class MaterialAnalyzer:
    """Moves ranked by material won against best replies; value = heuristic balance."""

    name = "material"

    def __init__(self, plies: int = 1):
        self.plies = plies
        self.oracle = Oracle()
        self.evaluator = HeuristicEvaluator()

    def analyze(self, game: ChessGame) -> PositionAnalysis | None:
        gains = self.oracle.material_gains(game, self.plies)
        if not gains:
            return None
        ranked = sorted(gains.items(), key=lambda item: item[1], reverse=True)
        top = max(-_CLIP, min(_CLIP, ranked[0][1]))
        # Softmax weights relative to the best move: best = 1.0, a pawn worse = 1/e.
        candidates = tuple(
            MoveInsight(move=move, score=math.exp(max(-_CLIP, min(_CLIP, gain)) - top))
            for move, gain in ranked
        )
        return PositionAnalysis(
            candidates=candidates, value=self.evaluator.evaluate(game), source=self.name
        )


__all__ = ["MaterialAnalyzer"]
