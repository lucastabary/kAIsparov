"""Tests for the `kaisparov train` CLI, especially config chaining.

These stay torch-free: `Trainer` and `build_config` are patched out so we only
exercise the argument parsing and the stage-to-stage resume threading.
"""

from __future__ import annotations

from textwrap import dedent
from unittest import mock

import pytest

from kaisparov import train
from kaisparov.training import chain
from kaisparov.training.config import load_train_config


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


# --------------------------------------------------------------- chain entry points
# A "chain config" is a YAML holding nothing but `stages:` — an entry point that
# expands to the very same stage list you'd have typed by hand.


def _write(path, text: str):
    path.write_text(dedent(text), encoding="utf-8")
    return path


def test_plain_config_passes_through_unexpanded(tmp_path):
    stage = _write(tmp_path / "one.yaml", "epochs: 3\n")
    assert chain.expand_config_chain([str(stage)]) == [str(stage)]


def test_chain_expands_to_its_stages_in_order(tmp_path):
    for name in ("a.yaml", "b.yaml", "c.yaml"):
        _write(tmp_path / name, "epochs: 1\n")
    entry = _write(tmp_path / "all.yaml", "stages: [a.yaml, b.yaml, c.yaml]\n")
    assert chain.expand_config_chain([str(entry)]) == [
        str(tmp_path / name) for name in ("a.yaml", "b.yaml", "c.yaml")
    ]


def test_chain_entries_may_be_globs(tmp_path):
    for name in ("phase1-1.yaml", "phase1-2.yaml", "phase2.yaml", "other.yaml"):
        _write(tmp_path / name, "epochs: 1\n")
    entry = _write(tmp_path / "all.yaml", "stages: ['phase*.yaml']\n")
    assert chain.expand_config_chain([str(entry)]) == [
        str(tmp_path / name) for name in ("phase1-1.yaml", "phase1-2.yaml", "phase2.yaml")
    ]


def test_chain_may_nest_and_mixes_with_plain_configs(tmp_path):
    _write(tmp_path / "a.yaml", "epochs: 1\n")
    _write(tmp_path / "b.yaml", "epochs: 1\n")
    inner = _write(tmp_path / "inner.yaml", "stages: [a.yaml, b.yaml]\n")
    outer = _write(tmp_path / "outer.yaml", "stages: [inner.yaml, a.yaml]\n")
    assert chain.expand_config_chain([str(outer), str(tmp_path / "b.yaml")]) == [
        str(tmp_path / "a.yaml"),
        str(tmp_path / "b.yaml"),
        str(tmp_path / "a.yaml"),
        str(tmp_path / "b.yaml"),
    ]
    assert chain.expand_config_chain([str(inner)])  # sanity: the inner one runs alone too


def test_chain_loop_is_refused(tmp_path):
    _write(tmp_path / "a.yaml", "stages: [b.yaml]\n")
    _write(tmp_path / "b.yaml", "stages: [a.yaml]\n")
    with pytest.raises(SystemExit, match="loops back"):
        chain.expand_config_chain([str(tmp_path / "a.yaml")])


def test_chain_rejects_training_settings(tmp_path):
    _write(tmp_path / "a.yaml", "epochs: 1\n")
    entry = _write(tmp_path / "all.yaml", "stages: [a.yaml]\nepochs: 10\n")
    with pytest.raises(SystemExit, match="epochs"):
        chain.expand_config_chain([str(entry)])


def test_missing_stage_names_the_chain(tmp_path):
    entry = _write(tmp_path / "all.yaml", "stages: [nope.yaml]\n")
    with pytest.raises(SystemExit, match="nope.yaml"):
        chain.expand_config_chain([str(entry)])


def test_training_a_chain_runs_every_stage_and_threads_run_ids(tmp_path):
    for name in ("a.yaml", "b.yaml"):
        _write(tmp_path / name, "epochs: 1\n")
    entry = _write(tmp_path / "all.yaml", "stages: [a.yaml, b.yaml]\n")
    assert _run_main(["--config", str(entry)]) == [
        (str(tmp_path / "a.yaml"), None),
        (str(tmp_path / "b.yaml"), "run1"),
    ]


def test_a_chain_is_not_a_training_config(tmp_path):
    entry = _write(tmp_path / "all.yaml", "stages: [a.yaml]\n")
    with pytest.raises(SystemExit, match="chain config"):
        load_train_config(str(entry))
