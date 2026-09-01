"""Tests for the run tracking + registry."""

from __future__ import annotations

from torch import nn

from kaisparov.tracking.registry import Registry
from kaisparov.tracking.run import RunManager


def test_runmanager_writes_artifacts_and_registry_reads_them(tmp_path):
    model = nn.Linear(3, 1)
    rm = RunManager(
        root=tmp_path, model="toy", config={"lr": 0.1}, seed=0, device="cpu", num_params=4
    )
    rm.log_metrics(1, {"loss": 0.5})
    rm.log_eval(2, {"elo_vs_random": 10.0})
    rm.save_checkpoint(model, 2, {"elo_vs_random": 10.0}, best_metric="elo_vs_random")
    rm.log_eval(4, {"elo_vs_random": 30.0})
    rm.save_checkpoint(model, 4, {"elo_vs_random": 30.0}, best_metric="elo_vs_random")
    rm.finish(final_metrics={"loss": 0.1})

    # Artifacts on disk
    assert (rm.dir / "config.yaml").exists()
    assert (rm.dir / "metrics.jsonl").exists()
    assert (rm.dir / "checkpoints" / "epoch4.pth").exists()
    assert (rm.dir / "checkpoints" / "best.pth").exists()

    # Registry round-trip
    reg = Registry(tmp_path)
    assert len(reg.list_runs()) == 1

    run = reg.get(rm.run_id)
    assert run["status"] == "completed"
    assert run["num_params"] == 4
    assert run["best_checkpoint"]["epoch"] == 4  # 30 > 10

    best_run, best_value = reg.best_run("elo_vs_random")
    assert best_value == 30.0
    assert best_run["run_id"] == rm.run_id

    assert reg.resolve_checkpoint(rm.run_id).name == "epoch4.pth"  # latest by default
    assert reg.resolve_checkpoint(rm.run_id, "best").name == "epoch4.pth"
    assert reg.resolve_checkpoint(rm.run_id, 2).name == "epoch2.pth"


def test_context_manager_marks_failed_on_exception(tmp_path):
    class Boom(RuntimeError):
        pass

    rm = RunManager(root=tmp_path, model="toy", config={}, seed=0, device="cpu")
    run_id = rm.run_id
    try:
        with rm:
            raise Boom()
    except Boom:
        pass

    assert Registry(tmp_path).get(run_id)["status"] == "failed"


def test_lineage_follows_parent_chain(tmp_path):
    parent = RunManager(root=tmp_path, model="toy", config={}, seed=0, device="cpu")
    parent.finish()
    child = RunManager(
        root=tmp_path,
        model="toy",
        config={},
        seed=0,
        device="cpu",
        parent_run_id=parent.run_id,
        resumed_from="somewhere/best.pth",
    )
    child.finish()

    chain = [r["run_id"] for r in Registry(tmp_path).lineage(child.run_id)]
    assert chain == [parent.run_id, child.run_id]
