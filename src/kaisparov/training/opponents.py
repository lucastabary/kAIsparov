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
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from kaisparov.agents.base import Policy


class OpponentPool:
    def __init__(
        self,
        spec,
        device: torch.device,
        hidden_dim: int,
        max_size: int = 5,
        seed=None,
        baselines: list | None = None,
        baseline_weight: float | None = None,
        snapshot_weight: float | None = None,
        baseline_weights: list | None = None,
        search_depth: int = 0,
    ):
        self.spec = spec
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        self.search_depth = search_depth
        # Group-level sampling weights (see `sample`); None -> legacy uniform draw.
        self.baseline_weight = baseline_weight
        self.snapshot_weight = snapshot_weight
        self._rng = random.Random(seed)
        self._agents: list = []
        # Fixed opponents (baselines) always available for sampling.
        self._baselines: list = list(baselines or [])
        # Per-baseline relative weights (same order as `baselines`); None = uniform.
        weights = list(baseline_weights or [])
        if weights and len(weights) != len(self._baselines):
            raise ValueError(
                f"baseline_weights has {len(weights)} entries but there are "
                f"{len(self._baselines)} baselines."
            )
        self._baseline_weights = weights or None

    def __len__(self) -> int:
        return len(self._agents) + len(self._baselines)

    def snapshot(self, model: torch.nn.Module) -> None:
        """Freeze a copy of the current weights and add it to the pool.

        With ``search_depth >= 1`` the frozen weights are wrapped in a Minimax
        search (a past self that *looks ahead* and refutes one-move blunders);
        otherwise they play as a plain sampling ``NeuralAgent``.
        """
        frozen = self.spec.model_class.create_agent(device=self.device, hidden_dim=self.hidden_dim)
        frozen.load_state_dict(copy.deepcopy(model.state_dict()))
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad_(False)
        processor = self.spec.processor_class()
        agent: Policy
        if self.search_depth >= 1:
            from kaisparov.agents.minimax_agent import MinimaxAgent

            agent = MinimaxAgent(frozen, processor, depth=self.search_depth)
        else:
            from kaisparov.agents.neural_agent import NeuralAgent

            # deterministic=False -> a bit of variety in the opponents' play.
            agent = NeuralAgent(frozen, processor, deterministic=False)
        self._agents.append(agent)
        if len(self._agents) > self.max_size:
            self._agents.pop(0)

    def sample(self):
        """Draw an opponent for one episode.

        Legacy (default, both group weights None): uniform over ``_agents +
        _baselines`` — but accumulating snapshots then drown out the baselines, so
        the learner rarely faces the opponents that punish tactical blunders. Set
        ``baseline_weight``/``snapshot_weight`` to give the two groups a *fixed*
        relative share regardless of how many snapshots exist, and
        ``baseline_weights`` to weight individual baselines within their group.
        """
        have_base = bool(self._baselines)
        have_snap = bool(self._agents)
        if not (have_base or have_snap):
            return None

        grouped = self.baseline_weight is not None or self.snapshot_weight is not None
        if grouped and have_base and have_snap:
            bw = self.baseline_weight if self.baseline_weight is not None else 1.0
            sw = self.snapshot_weight if self.snapshot_weight is not None else 1.0
            if self._rng.choices((True, False), weights=(bw, sw))[0]:
                return self._rng.choices(self._baselines, weights=self._baseline_weights)[0]
            return self._rng.choice(self._agents)

        # A single non-empty group, or the legacy uniform draw over the union.
        if have_base and not have_snap:
            return self._rng.choices(self._baselines, weights=self._baseline_weights)[0]
        if have_snap and not have_base:
            return self._rng.choice(self._agents)
        return self._rng.choice(self._agents + self._baselines)
