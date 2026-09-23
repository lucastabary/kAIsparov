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

from kaisparov.models.architecture import Architecture

if TYPE_CHECKING:
    from kaisparov.agents.base import Policy


# Stands for "a past-self, drawn uniformly" in the weighted draw of OpponentPool.sample.
_STREAM = object()


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
        architecture: Architecture,
        device: torch.device,
        max_size: int = 5,
        seed=None,
        baselines: list | None = None,
        baseline_weights: list | None = None,
        snapshot_weight: float | None = None,
        search_depth: int = 0,
        avoid_king_suicide: bool = False,
        snapshot_deterministic: bool = False,
        snapshot_random_move_prob: float = 0.0,
    ):
        # The learner's: every past-self is a frozen copy of the learner.
        self.architecture = architecture
        self.device = device
        self.max_size = max_size
        self.search_depth = search_depth
        # Past-selves added to the pool refuse moves that walk into mate in one.
        self.avoid_king_suicide = avoid_king_suicide
        # Only used by depth-0 snapshots (raw NeuralAgent): sample vs argmax.
        self.snapshot_deterministic = snapshot_deterministic
        # Chance, on each move, that a past-self plays a random legal move (Fallible).
        self.snapshot_random_move_prob = snapshot_random_move_prob
        # Share of the whole past-self stream, on the same scale as `baseline_weights`
        # (see `sample`). None -> the legacy uniform draw over baselines + snapshots.
        self.snapshot_weight = snapshot_weight
        self._rng = random.Random(seed)
        self._agents: list = []
        # CPU state_dict of each snapshot's frozen model, kept in lock-step with
        # ``_agents`` so the pool can be shipped to / rebuilt in worker processes.
        self._snapshot_sds: list[dict] = []
        # Fixed opponents (baselines) always available for sampling.
        self._baselines: list = list(baselines or [])
        # Per-baseline weights (same order as `baselines`); None = 1 each.
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
        otherwise they play as a plain sampling ``NeuralAgent``. Either way it plays
        a random move with probability ``snapshot_random_move_prob``.
        """
        from kaisparov.models.factory import build_agent

        frozen, processor = build_agent(self.architecture, self.device)
        frozen.load_state_dict(state_dict)
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad_(False)
        agent: Policy
        if self.search_depth >= 1:
            from kaisparov.agents.minimax_agent import MinimaxAgent

            agent = MinimaxAgent.on_model(
                frozen,
                processor,
                depth=self.search_depth,
                avoid_king_suicide=self.avoid_king_suicide,
            )
        else:
            from kaisparov.agents.neural_agent import NeuralAgent

            # deterministic=False -> a bit of variety in the opponents' play.
            agent = NeuralAgent(
                frozen,
                processor,
                deterministic=self.snapshot_deterministic,
                avoid_king_suicide=self.avoid_king_suicide,
            )
        return _fallible(agent, self.snapshot_random_move_prob, self._rng.randrange(2**31))

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

        Every baseline, and the past-self stream as a whole, has a weight on one scale:
        an opponent is drawn with probability ``weight / sum(weights)``, so out of N
        games each gets ``N * weight / sum`` of them, whatever the number of snapshots
        (within the stream, the snapshots are drawn uniformly). Until the first
        snapshot exists, the stream's share goes to the baselines.

        With ``snapshot_weight`` None (the legacy flat fields, without group weights)
        the draw is uniform over baselines + snapshots — accumulating snapshots then
        drown out the baselines.
        """
        have_snap = bool(self._agents)
        if not (self._baselines or have_snap):
            return None
        if self.snapshot_weight is None and have_snap:
            return self._rng.choice(self._agents + self._baselines)

        entries: list = list(self._baselines)
        weights = list(self._baseline_weights or [1.0] * len(self._baselines))
        if have_snap:
            entries.append(_STREAM)
            weights.append(float(self.snapshot_weight or 0.0))
        if sum(weights) <= 0:
            weights = [1.0] * len(entries)  # every weight zero: fall back to uniform
        picked = self._rng.choices(entries, weights=weights)[0]
        return self._rng.choice(self._agents) if picked is _STREAM else picked


def _fallible(agent, random_move_prob: float, seed: int):
    """``agent``, playing a random move with probability ``random_move_prob`` (0: as is)."""
    if not random_move_prob:
        return agent
    from kaisparov.agents.fallible import Fallible

    return Fallible(agent, random_move_prob, seed=seed)


def build_pool_baseline(device, opp, seed: int):
    """Build one fixed baseline agent from a pool-preset entry (:class:`OpponentSpec`).

    ``random``/``material`` are model-free; so is ``minimax`` with ``params.evaluator``
    (``material`` or ``heuristic``: an alpha-beta search on it, ``depth`` plies).
    ``neural``/``minimax`` otherwise load a frozen model from ``params.checkpoint`` (a
    past run's weights) — a strong, fixed teacher. It is rebuilt as the architecture
    *its* run recorded, which need not be the learner's: each agent graphifies the
    board with its own processor. Any of them plays a random move with probability
    ``opp.random_move_prob``.
    """
    return _fallible(_build_pool_agent(device, opp, seed), opp.random_move_prob, seed)


def _build_pool_agent(device, opp, seed: int):
    params, kind = opp.params, opp.kind
    if kind in ("random", "material"):
        return build_baseline(
            kind, params.get("seed", seed), params.get("avoid_king_suicide", False)
        )
    if kind == "minimax" and params.get("evaluator"):
        from kaisparov.agents.minimax_agent import MinimaxAgent
        from kaisparov.analysis.evaluators import HeuristicEvaluator, MaterialEvaluator

        evaluators: dict[str, type[MaterialEvaluator] | type[HeuristicEvaluator]] = {
            "material": MaterialEvaluator,
            "heuristic": HeuristicEvaluator,
        }
        return MinimaxAgent(
            evaluators[params["evaluator"]](),
            depth=int(params.get("depth", 2)),
            seed=params.get("seed", seed),
            avoid_king_suicide=params.get("avoid_king_suicide", False),
        )
    if kind in ("neural", "minimax"):
        checkpoint = params.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"A '{kind}' baseline needs params.checkpoint (frozen weights path).")
        from kaisparov.models.factory import load_agent

        loaded = load_agent(checkpoint, device, frozen=True)
        model, processor = loaded.model, loaded.processor
        if kind == "minimax":
            from kaisparov.agents.minimax_agent import MinimaxAgent

            return MinimaxAgent.on_model(
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


def build_opponent_pool(architecture: Architecture, device, rollout, seed: int):
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
            baselines.append(build_pool_baseline(device, opp, seed))
            baseline_weights.append(opp.weight)
        snap = pspec.snapshot
        sp = snap.params if snap else {}
        pool = OpponentPool(
            architecture,
            device,
            max_size=snap.count if snap else 0,
            seed=seed,
            baselines=baselines,
            baseline_weights=baseline_weights or None,
            snapshot_weight=snap.weight if snap else None,
            search_depth=int(sp.get("depth", 0)),
            avoid_king_suicide=bool(sp.get("avoid_king_suicide", False)),
            snapshot_deterministic=bool(sp.get("deterministic", False)),
            snapshot_random_move_prob=snap.random_move_prob if snap else 0.0,
        )
        return pool, (snap is not None), (snap.every if snap else rollout.snapshot_every)

    # Flat legacy path: baselines named in rollout.baselines, with the old two-level
    # weights (a share per group, then per baseline within its group) flattened onto
    # the pool's single scale: baseline i gets group_share * w_i / sum(w).
    avoid = rollout.opponent_avoid_king_suicide
    baselines = [build_baseline(n, seed, avoid_king_suicide=avoid) for n in rollout.baselines]
    weights: list[float] | None = list(rollout.baseline_weights) or None
    snapshot_weight: float | None = None
    if rollout.baseline_weight is not None or rollout.snapshot_weight is not None:
        group = rollout.baseline_weight if rollout.baseline_weight is not None else 1.0
        within = weights or [1.0] * len(baselines)
        total = sum(within) or 1.0
        weights = [group * w / total for w in within]
        snapshot_weight = rollout.snapshot_weight if rollout.snapshot_weight is not None else 1.0
    pool = OpponentPool(
        architecture,
        device,
        max_size=rollout.pool_size,
        seed=seed,
        baselines=baselines,
        baseline_weights=weights,
        snapshot_weight=snapshot_weight,
        search_depth=rollout.snapshot_search_depth,
        avoid_king_suicide=avoid,
    )
    return pool, True, rollout.snapshot_every
