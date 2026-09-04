"""Unit tests for the parallel-rollout helpers (episode split / worker resolution).

The end-to-end multiprocess path (spawning workers, merging buffers, training on
the result) is exercised by the standalone benchmark, not here, to keep the test
suite fast and free of process spawning.
"""

from __future__ import annotations

from kaisparov.training.parallel_rollout import _split_episodes, resolve_num_workers


def test_split_episodes_sums_to_total_and_is_balanced():
    counts = _split_episodes(64, 12)
    assert sum(counts) == 64
    assert len(counts) == 12
    assert max(counts) - min(counts) <= 1  # as even as possible


def test_split_episodes_drops_idle_workers():
    # More workers than episodes: only as many chunks as there are episodes.
    assert _split_episodes(3, 8) == [1, 1, 1]


def test_split_episodes_one_worker_gets_everything():
    assert _split_episodes(10, 1) == [10]


def test_resolve_num_workers_auto_and_floor():
    assert resolve_num_workers(0) >= 1  # 0 -> os.cpu_count()
    assert resolve_num_workers(4) == 4
    assert resolve_num_workers(1) == 1
    assert resolve_num_workers(-3) == 1  # floored at 1
