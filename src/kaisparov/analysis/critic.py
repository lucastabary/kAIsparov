"""Use a trained model's value head as the evaluator behind the move review.

This is the interesting one for the project: instead of grading moves against
material, grade them against what the GNN *believes* about the position. It shares
the loaded model and processor with the agent and the analyzer, so it costs one
forward pass per evaluated position and nothing extra to set up.

Caveat worth knowing before reading a verdict: the critic is trained on shaped
self-play returns, so its output is in reward units, not pawns — ``slope`` is the
knob that maps those units onto winning chances, and it is a calibration guess, not
a measurement. Material is the better-calibrated default; this one is the honest
"what does the model think" view.
"""

from __future__ import annotations

import torch

from kaisparov.core.board import ChessGame


class CriticEvaluator:
    """Evaluator backed by an actor-critic's value head (side-to-move point of view)."""

    name = "critic"
    default_lookahead = 0  # a forward pass per node: one ply of candidates is enough

    def __init__(self, model: torch.nn.Module, processor, slope: float = 1.5):
        self.model = model
        self.processor = processor
        self.slope = slope
        self.model.eval()

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def evaluate(self, game: ChessGame) -> float:
        with torch.no_grad():
            data = self.processor.graphify(game).to(self._device())
            _, value = self.model(data)
            return float(value.reshape(-1)[0])


__all__ = ["CriticEvaluator"]
