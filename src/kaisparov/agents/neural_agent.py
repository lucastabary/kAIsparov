from __future__ import annotations

import torch

from kaisparov.models.base_processor import ModelAction


class NeuralAgent:
    """Thin inference wrapper around a model and its output processor.

    Implements the :class:`~kaisparov.agents.base.Policy` interface via
    :meth:`select_move`, so it can face the baselines in the arena.
    """

    name = "neural"

    def __init__(
        self,
        model: torch.nn.Module,
        processor,
        deterministic: bool = False,
        avoid_king_suicide: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.deterministic = deterministic
        # When True, mask out moves that hang our own king so the policy only samples
        # king-safe moves (unless none exist). Best-effort: requires the processor to
        # expose ``move_mask``; otherwise it's a no-op. See kaisparov.agents.safety.
        self.avoid_king_suicide = avoid_king_suicide

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
                legal_mask=self._king_safe_mask(game),
            )

    def _king_safe_mask(self, game):
        """A ``legal_mask`` restricted to king-safe moves, or ``None`` to use the
        default legal mask (guard off, no unsafe move to drop, or processor can't
        build one)."""
        if not self.avoid_king_suicide or not hasattr(self.processor, "move_mask"):
            return None
        from kaisparov.agents.safety import safe_moves
        from kaisparov.core.movegen import all_moves

        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        if not moves:
            return None
        safe = safe_moves(game, moves)
        if len(safe) == len(moves):
            return None  # nothing to restrict -> default legal mask
        return self.processor.move_mask(game, safe)
