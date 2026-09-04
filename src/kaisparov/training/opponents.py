"""Opponent pool (a small "league") for training against a varied roster.

Playing self-play against the *current* policy can collapse into a degenerate
equilibrium (everyone rushes). Facing a pool of opponents breaks that collapse and,
crucially, lets us seed *fixed baselines* (``RandomAgent``, ``MaterialAgent``) that
punish hung pieces and king exposure from epoch 1 — long before any self-snapshot
exists. The pool mixes these baselines with frozen past ``NeuralAgent`` checkpoints.
"""

from __future__ import annotations

import copy
import random

import torch


class OpponentPool:
    def __init__(
        self,
        spec,
        device: torch.device,
        hidden_dim: int,
        max_size: int = 5,
        seed=None,
        baselines: list | None = None,
    ):
        self.spec = spec
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        self._rng = random.Random(seed)
        self._agents: list = []
        # Fixed opponents (baselines) always available for sampling.
        self._baselines: list = list(baselines or [])

    def __len__(self) -> int:
        return len(self._agents) + len(self._baselines)

    def snapshot(self, model: torch.nn.Module) -> None:
        """Freeze a copy of the current weights and add it to the pool."""
        from kaisparov.agents.neural_agent import NeuralAgent

        frozen = self.spec.model_class.create_agent(device=self.device, hidden_dim=self.hidden_dim)
        frozen.load_state_dict(copy.deepcopy(model.state_dict()))
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad_(False)
        # deterministic=False -> a bit of variety in the opponents' play.
        self._agents.append(NeuralAgent(frozen, self.spec.processor_class(), deterministic=False))
        if len(self._agents) > self.max_size:
            self._agents.pop(0)

    def sample(self):
        """A random opponent (baseline or frozen snapshot), or None if empty."""
        choices = self._agents + self._baselines
        return self._rng.choice(choices) if choices else None
