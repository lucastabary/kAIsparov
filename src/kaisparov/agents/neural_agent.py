from __future__ import annotations

import torch

from kaisparov.models.base_processor import ModelAction


class NeuralAgent:
    """Thin inference wrapper around a model and its output processor.

    Implements the :class:`~kaisparov.agents.base.Policy` interface via
    :meth:`select_move`, so it can face the baselines in the arena.
    """

    name = "neural"

    def __init__(self, model: torch.nn.Module, processor, deterministic: bool = False):
        self.model = model
        self.processor = processor
        self.deterministic = deterministic

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def select_move(self, game):
        """Return the chosen ``(source, dest)`` move, or ``None`` if the model has
        no legal action for the current state."""
        try:
            return self.act(game).move_coords
        except RuntimeError:
            return None

    def act(self, game) -> ModelAction:
        self.model.eval()
        with torch.no_grad():
            state_data = self.processor.graphify(game)
            model_output = self.model(state_data.to(self._model_device()))
            return self.processor.process_output(
                model_output=model_output,
                game=game,
                deterministic=self.deterministic,
            )
