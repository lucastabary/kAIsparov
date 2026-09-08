"""Opponent pool (a small "league") for training against a varied roster.

Playing self-play against the *current* policy can collapse into a degenerate
equilibrium (everyone rushes). Facing a pool of opponents breaks that collapse and,
crucially, lets us seed *fixed baselines* (``RandomAgent``, ``MaterialAgent``) that
punish hung pieces and king exposure from epoch 1 — long before any self-snapshot
exists. The pool mixes these baselines with frozen past ``NeuralAgent`` checkpoints.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from kaisparov.agents.base import Policy


def build_baseline(name: str, seed: int, avoid_king_suicide: bool = False):
    """Instantiate a fixed baseline opponent by name (shared by trainer and workers)."""
    if name == "random":
        from kaisparov.agents.random_agent import RandomAgent

        return RandomAgent(seed=seed, avoid_king_suicide=avoid_king_suicide)
    if name == "material":
        from kaisparov.agents.material_agent import MaterialAgent

        return MaterialAgent(seed=seed, avoid_king_suicide=avoid_king_suicide)
    raise ValueError(f"Unknown baseline opponent '{name}' (expected 'random' or 'material').")


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
        avoid_king_suicide: bool = False,
        snapshot_deterministic: bool = False,
    ):
        self.spec = spec
        self.device = device
        self.hidden_dim = hidden_dim
        self.max_size = max_size
        self.search_depth = search_depth
        # Past-selves added to the pool refuse moves that hang their own king.
        self.avoid_king_suicide = avoid_king_suicide
        # Only used by depth-0 snapshots (raw NeuralAgent): sample vs argmax.
        self.snapshot_deterministic = snapshot_deterministic
        # Group-level sampling weights (see `sample`); None -> legacy uniform draw.
        self.baseline_weight = baseline_weight
        self.snapshot_weight = snapshot_weight
        self._rng = random.Random(seed)
        self._agents: list = []
        # CPU state_dict of each snapshot's frozen model, kept in lock-step with
        # ``_agents`` so the pool can be shipped to / rebuilt in worker processes.
        self._snapshot_sds: list[dict] = []
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

    def _make_snapshot_agent(self, state_dict: dict) -> Policy:
        """Wrap a frozen copy of ``state_dict`` as a pool opponent.

        With ``search_depth >= 1`` the frozen weights are wrapped in a Minimax
        search (a past self that *looks ahead* and refutes one-move blunders);
        otherwise they play as a plain sampling ``NeuralAgent``.
        """
        frozen = self.spec.model_class.create_agent(device=self.device, hidden_dim=self.hidden_dim)
        frozen.load_state_dict(state_dict)
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad_(False)
        processor = self.spec.processor_class()
        if self.search_depth >= 1:
            from kaisparov.agents.minimax_agent import MinimaxAgent

            return MinimaxAgent(
                frozen,
                processor,
                depth=self.search_depth,
                avoid_king_suicide=self.avoid_king_suicide,
            )
        from kaisparov.agents.neural_agent import NeuralAgent

        # deterministic=False -> a bit of variety in the opponents' play.
        return NeuralAgent(
            frozen,
            processor,
            deterministic=self.snapshot_deterministic,
            avoid_king_suicide=self.avoid_king_suicide,
        )

    def _add_snapshot(self, agent: Policy, state_dict: dict) -> None:
        self._agents.append(agent)
        self._snapshot_sds.append(state_dict)
        if len(self._agents) > self.max_size:
            self._agents.pop(0)
            self._snapshot_sds.pop(0)

    def snapshot(self, model: torch.nn.Module) -> None:
        """Freeze a copy of the current weights and add it to the pool."""
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._add_snapshot(self._make_snapshot_agent(sd), sd)

    def add_snapshot_state_dict(self, state_dict: dict) -> None:
        """Add a snapshot from a (CPU) ``state_dict`` — used to rebuild the pool in workers."""
        self._add_snapshot(self._make_snapshot_agent(state_dict), state_dict)

    def snapshot_state_dicts(self) -> list[dict]:
        """CPU state_dicts of the current snapshots, in order (for shipping to workers)."""
        return list(self._snapshot_sds)

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


def _load_frozen_model(spec, device, hidden_dim, checkpoint: str):
    """Load a frozen (eval, no-grad) model of this architecture from a checkpoint."""
    model = spec.model_class.create_agent(device=device, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_pool_baseline(spec, device, hidden_dim, opp, seed: int):
    """Build one fixed baseline agent from a pool-preset entry (:class:`OpponentSpec`).

    ``random``/``material`` are model-free; ``neural``/``minimax`` load a frozen model
    from ``params.checkpoint`` (a past run's weights) — a strong, fixed teacher.
    """
    params, kind = opp.params, opp.kind
    if kind in ("random", "material"):
        return build_baseline(
            kind, params.get("seed", seed), params.get("avoid_king_suicide", False)
        )
    if kind in ("neural", "minimax"):
        checkpoint = params.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"A '{kind}' baseline needs params.checkpoint (frozen weights path).")
        model = _load_frozen_model(spec, device, hidden_dim, checkpoint)
        processor = spec.processor_class()
        if kind == "minimax":
            from kaisparov.agents.minimax_agent import MinimaxAgent

            return MinimaxAgent(
                model,
                processor,
                depth=int(params.get("depth", 2)),
                avoid_king_suicide=params.get("avoid_king_suicide", False),
            )
        from kaisparov.agents.neural_agent import NeuralAgent

        return NeuralAgent(
            model,
            processor,
            deterministic=params.get("deterministic", False),
            avoid_king_suicide=params.get("avoid_king_suicide", False),
        )
    raise ValueError(f"Kind {kind!r} is not valid for a fixed baseline (group: baseline).")


def build_opponent_pool(spec, device, hidden_dim: int, rollout, seed: int):
    """Build the opponent pool from a ``RolloutSettings`` — the single source of truth
    shared by the trainer and the parallel workers.

    Only the fixed baselines are built here; the accumulating past-selves (snapshots)
    are added at runtime via :meth:`OpponentPool.snapshot` /
    :meth:`OpponentPool.add_snapshot_state_dict`. Returns
    ``(pool, take_snapshots, snapshot_every)``.
    """
    if rollout.pool is not None:
        from kaisparov.training.config import build_pool_spec

        pspec = build_pool_spec(rollout.pool)
        baselines, baseline_weights = [], []
        for opp in pspec.baselines:
            baselines.append(build_pool_baseline(spec, device, hidden_dim, opp, seed))
            baseline_weights.append(opp.weight)
        snap = pspec.snapshot
        sp = snap.params if snap else {}
        gw = pspec.group_weights
        pool = OpponentPool(
            spec,
            device,
            hidden_dim,
            max_size=snap.count if snap else 0,
            seed=seed,
            baselines=baselines,
            baseline_weight=gw.get("baseline"),
            snapshot_weight=gw.get("snapshot"),
            baseline_weights=baseline_weights or None,
            search_depth=int(sp.get("depth", 0)),
            avoid_king_suicide=bool(sp.get("avoid_king_suicide", False)),
            snapshot_deterministic=bool(sp.get("deterministic", False)),
        )
        return pool, (snap is not None), (snap.every if snap else rollout.snapshot_every)

    # Flat legacy path: baselines named in rollout.baselines.
    avoid = rollout.opponent_avoid_king_suicide
    baselines = [build_baseline(n, seed, avoid_king_suicide=avoid) for n in rollout.baselines]
    pool = OpponentPool(
        spec,
        device,
        hidden_dim,
        max_size=rollout.pool_size,
        seed=seed,
        baselines=baselines,
        baseline_weight=rollout.baseline_weight,
        snapshot_weight=rollout.snapshot_weight,
        baseline_weights=rollout.baseline_weights,
        search_depth=rollout.snapshot_search_depth,
        avoid_king_suicide=avoid,
    )
    return pool, True, rollout.snapshot_every
