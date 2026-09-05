"""Tests for the `kaisparov train` CLI, especially config chaining.

These stay torch-free: `Trainer` and `build_config` are patched out so we only
exercise the argument parsing and the stage-to-stage resume threading.
"""

from __future__ import annotations

from unittest import mock

from kaisparov import train


def test_config_accepts_multiple_paths():
    assert train.parse_args(["--config", "a.yaml"]).config == ["a.yaml"]
    assert train.parse_args(["--config", "a.yaml", "b.yaml", "c.yaml"]).config == [
        "a.yaml",
        "b.yaml",
        "c.yaml",
    ]


class _FakeTrainer:
    """Returns a deterministic run id per stage without touching torch."""

    _n = 0

    def __init__(self, config):
        self.config = config

    def train(self) -> str:
        _FakeTrainer._n += 1
        return f"run{_FakeTrainer._n}"


def _run_main(argv):
    """Run train.main with Trainer/build_config stubbed; return build_config calls."""
    calls: list[tuple[str | None, str | None]] = []

    def fake_build_config(args, config_path=None, resume_run_id=None):
        calls.append((config_path, resume_run_id))
        return object()

    _FakeTrainer._n = 0
    with (
        mock.patch.object(train, "Trainer", _FakeTrainer),
        mock.patch.object(train, "build_config", fake_build_config),
    ):
        train.main(argv)
    return calls


def test_chain_threads_previous_run_id_into_next_stage():
    calls = _run_main(["--config", "s1.yaml", "s2.yaml", "s3.yaml"])
    # First stage starts fresh; each later stage resumes from the prior run's id.
    assert calls == [("s1.yaml", None), ("s2.yaml", "run1"), ("s3.yaml", "run2")]


def test_single_config_does_not_chain():
    assert _run_main(["--config", "only.yaml"]) == [("only.yaml", None)]


def test_no_config_runs_one_default_stage():
    assert _run_main([]) == [(None, None)]
