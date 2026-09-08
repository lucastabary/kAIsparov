"""Parallel rollout collection (multiprocessing) — smoke coverage.

Spawns worker processes, so it exercises the payload pickling and the in-worker pool
rebuild that the in-process tests can't. Kept tiny (2 workers, few short games).
"""

from __future__ import annotations

import torch

from kaisparov.models.factory import load_backend_spec
from kaisparov.training.config import RewardSettings, RolloutSettings
from kaisparov.training.parallel_rollout import (
    collect_vs_opponent_parallel,
    shutdown_executor,
)

_VS_KEYS = ("king_capture_rate", "winrate", "lossrate", "drawrate", "avg_plies", "transitions")


def test_collect_vs_opponent_parallel_fills_buffer():
    spec = load_backend_spec("rgcn")
    agent = spec.model_class.create_agent(device=torch.device("cpu"), hidden_dim=4)
    buffer = spec.buffer_class(gamma=0.99, gae_lambda=0.95, self_play=False)
    rollout = RolloutSettings(
        opponent="pool",
        num_workers=2,
        episodes_per_epoch=4,
        max_steps_per_episode=8,
        pool={
            "opponents": [
                {"kind": "material", "group": "baseline"},
                {"kind": "random", "group": "baseline"},
            ]
        },
    )
    try:
        stats = collect_vs_opponent_parallel(
            agent,
            buffer,
            num_workers=2,
            num_episodes=4,
            max_steps_per_episode=8,
            model_name="rgcn",
            hidden_dim=4,
            reward_settings=RewardSettings(),
            curriculum_settings=None,
            gamma=0.99,
            gae_lambda=0.95,
            base_seed=1,
            rollout=rollout,
            snapshot_sds=[],
        )
    finally:
        shutdown_executor()

    assert len(buffer) > 0
    assert stats["transitions"] == float(len(buffer))
    assert all(key in stats for key in _VS_KEYS)
