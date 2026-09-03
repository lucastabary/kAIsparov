"""Tests for the git-log-style lineage view (torch-free)."""

from __future__ import annotations

from kaisparov.tracking.lineage_view import _assign_lanes, _order, build_html


def _run(rid: str, parent: str | None = None, **extra):
    base = {
        "run_id": rid,
        "created_at": f"2026-09-03T00:00:{rid[-2:]}+00:00",
        "parent_run_id": parent,
        "status": "completed",
        "model": "rgcn",
        "config": {"ppo": {"learning_rate": 0.001}, "seed": 0},
    }
    base.update(extra)
    return base


def test_order_newest_first():
    runs = [_run("a01"), _run("a03"), _run("a02")]
    assert [r["run_id"] for r in _order(runs)] == ["a03", "a02", "a01"]


def test_linear_chain_shares_one_lane():
    # a01 -> a02 -> a03 (each resumes the previous): a single family, one lane.
    runs = [_run("a01"), _run("a02", "a01"), _run("a03", "a02")]
    lanes = _assign_lanes(_order(runs))
    assert lanes == {"a01": 0, "a02": 0, "a03": 0}


def test_fork_uses_a_second_lane():
    # a02 and a03 both resume a01 -> a01 keeps lane 0, the branch takes another lane.
    runs = [_run("a01"), _run("a02", "a01"), _run("a03", "a01")]
    lanes = _assign_lanes(_order(runs))
    assert lanes["a01"] == 0
    assert lanes["a02"] != lanes["a03"]  # the two branches occupy distinct lanes


def test_overlapping_families_get_distinct_lanes():
    # a01 is still "open" (a03 resumes it later) when unrelated root b02 appears,
    # so b02 must not land in a01's lane.
    runs = [_run("a01"), _run("b02"), _run("a03", "a01")]
    lanes = _assign_lanes(_order(runs))
    assert lanes["a01"] == lanes["a03"]
    assert lanes["b02"] != lanes["a01"]


def test_build_html_is_self_contained_and_marks_diffs():
    parent = _run("a01", title="root", config={"ppo": {"learning_rate": 0.001}})
    child = _run("a02", "a01", title="resume", config={"ppo": {"learning_rate": 0.005}})
    html = build_html([parent, child], root="runs")
    assert "<!doctype html>" in html
    assert "http" not in html.split("<style>")[0].lower() or "://" not in html  # no external assets
    assert "run lineage" in html
    assert "changed" in html  # the learning-rate change is highlighted
    assert "was 0.001" in html  # old value shown like a diff
    assert "a01" in html and "a02" in html


def test_build_html_empty():
    html = build_html([], root="runs")
    assert "No runs found" in html
