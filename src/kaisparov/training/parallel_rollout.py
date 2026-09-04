"""Data-parallel self-play collection.

The self-play rollout (:func:`kaisparov.training.rollout.collect_data`) already
batches the model forward across an epoch's episodes, so its remaining cost is the
pure-Python engine work (graphify, legal-mask, make/unmake) run per game per ply on
one core. This module spreads that work across processes: each worker rebuilds the
agent from a CPU ``state_dict``, plays a slice of the epoch's episodes into its own
buffer, and the parent concatenates the slices back into one buffer.

Episodes are independent and each is flushed with a terminal ``done`` flag, so the
negamax GAE in :mod:`kaisparov.training.ppo` — which resets at every ``done`` — is
unaffected by how episodes are distributed or concatenated.

Only self-play is parallelised here; league/pool collection keeps live opponent
objects that are not cheap to ship to workers, so it stays in-process.

Note: workers use spawn and their own RNG streams, so a parallel run is not
bit-identical to the in-process one — expected for RL, and each worker is seeded
distinctly so the slices are decorrelated.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any

import torch

from kaisparov.core.board import ChessGame
from kaisparov.training.config import CurriculumSettings, RewardSettings

# One persistent pool, reused across epochs so torch is imported once per worker
# instead of once per collection. Re-created if the requested worker count changes.
_EXECUTOR: ProcessPoolExecutor | None = None
_EXECUTOR_WORKERS: int | None = None


def _init_worker() -> None:
    # Each process plays on one core; without this, N processes x M intra-op threads
    # oversubscribe the CPU and the parallel run can be slower than the serial one.
    torch.set_num_threads(1)


def _get_executor(num_workers: int) -> ProcessPoolExecutor:
    global _EXECUTOR, _EXECUTOR_WORKERS
    if _EXECUTOR is None or num_workers != _EXECUTOR_WORKERS:
        shutdown_executor()
        _EXECUTOR = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
        )
        _EXECUTOR_WORKERS = num_workers
    return _EXECUTOR


def shutdown_executor() -> None:
    """Tear down the persistent worker pool (idempotent)."""
    global _EXECUTOR, _EXECUTOR_WORKERS
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown(wait=False)
    _EXECUTOR = None
    _EXECUTOR_WORKERS = None


atexit.register(shutdown_executor)


def resolve_num_workers(num_workers: int) -> int:
    """0 -> os.cpu_count(); otherwise the value itself (floored at 1)."""
    if num_workers == 0:
        return max(os.cpu_count() or 1, 1)
    return max(num_workers, 1)


def _split_episodes(num_episodes: int, num_workers: int) -> list[int]:
    """Distribute episodes as evenly as possible; drop workers that get none."""
    base, rem = divmod(num_episodes, num_workers)
    counts = [base + (1 if i < rem else 0) for i in range(num_workers)]
    return [c for c in counts if c > 0]


def _worker_collect(payload: dict[str, Any]) -> dict[str, Any]:
    """Play ``payload['num_episodes']`` self-play games and return raw transitions."""
    import random

    import numpy as np

    from kaisparov.models.factory import load_backend, load_backend_spec
    from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum
    from kaisparov.training.reward import make_reward_fn

    seed = payload["seed"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)

    model_name = payload["model_name"]
    spec = load_backend_spec(model_name)
    module = load_backend(model_name)

    agent = spec.model_class.create_agent(
        device=torch.device("cpu"), hidden_dim=payload["hidden_dim"]
    )
    agent.load_state_dict(payload["state_dict"])
    agent.eval()

    reward_fn = make_reward_fn(RewardSettings(**payload["reward"]))

    curriculum = None
    if payload["curriculum"] is not None:
        curriculum = PieceCountCurriculum(PhaseConfig(**payload["curriculum"]), seed=seed)

    buffer = spec.buffer_class(
        gamma=payload["gamma"], gae_lambda=payload["gae_lambda"], self_play=payload["self_play"]
    )
    stats = spec.collect_data(
        agent,
        ChessGame(),
        buffer,
        num_episodes=payload["num_episodes"],
        max_steps_per_episode=payload["max_steps_per_episode"],
        model_module=module,
        curriculum=curriculum,
        reward_fn=reward_fn,
    )
    return {
        "states": buffer.states,
        "actions": buffer.actions,
        "log_probs": buffer.log_probs,
        "values": buffer.values,
        "rewards": buffer.rewards,
        "dones": buffer.dones,
        "legal_masks": buffer.legal_masks,
        "num_episodes": payload["num_episodes"],
        "stats": stats,
    }


# Keys collect_data reports as per-episode rates/means; aggregated as an
# episode-weighted average across worker slices.
_WEIGHTED_KEYS = ("king_capture_rate", "truncated_rate", "stalemate_rate", "avg_plies")


def collect_data_parallel(
    agent: torch.nn.Module,
    buffer,
    *,
    num_workers: int,
    num_episodes: int,
    max_steps_per_episode: int,
    model_name: str,
    hidden_dim: int,
    reward_settings: RewardSettings,
    curriculum_settings: CurriculumSettings | None,
    gamma: float,
    gae_lambda: float,
    self_play: bool,
    base_seed: int,
) -> dict[str, float]:
    """Collect ``num_episodes`` self-play games across processes into ``buffer``.

    ``buffer`` is extended in place with the concatenated transitions; the return
    value mirrors :func:`kaisparov.training.rollout.collect_data`'s stats dict.
    """
    workers = resolve_num_workers(num_workers)
    counts = _split_episodes(num_episodes, workers)
    if not counts:
        return {"transitions": float(len(buffer))}

    state_dict = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
    reward = asdict(reward_settings)
    curriculum = asdict(curriculum_settings) if curriculum_settings is not None else None

    payloads = [
        {
            "num_episodes": count,
            "max_steps_per_episode": max_steps_per_episode,
            "model_name": model_name,
            "hidden_dim": hidden_dim,
            "state_dict": state_dict,
            "reward": reward,
            "curriculum": curriculum,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "self_play": self_play,
            "seed": base_seed * 100003 + i,  # decorrelate worker RNG streams
        }
        for i, count in enumerate(counts)
    ]

    executor = _get_executor(len(counts))
    results = list(executor.map(_worker_collect, payloads))

    for r in results:
        buffer.states.extend(r["states"])
        buffer.actions.extend(r["actions"])
        buffer.log_probs.extend(r["log_probs"])
        buffer.values.extend(r["values"])
        buffer.rewards.extend(r["rewards"])
        buffer.dones.extend(r["dones"])
        buffer.legal_masks.extend(r["legal_masks"])

    total = sum(r["num_episodes"] for r in results) or 1
    stats = {
        key: sum(r["stats"].get(key, 0.0) * r["num_episodes"] for r in results) / total
        for key in _WEIGHTED_KEYS
    }
    stats["transitions"] = float(len(buffer))
    return stats


__all__ = ["collect_data_parallel", "resolve_num_workers", "shutdown_executor"]
