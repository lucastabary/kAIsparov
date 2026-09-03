"""Registry — query the runs recorded by :class:`RunManager`.

Read-only and torch-free, so ``python -m kaisparov.tracking`` stays fast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class Registry:
    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)

    def _run_dirs(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [d for d in self.root.iterdir() if (d / "run.json").exists()]

    def get(self, run_id: str) -> dict[str, Any]:
        path = self.root / run_id / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"No run '{run_id}' under {self.root}")
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for d in self._run_dirs():
            with (d / "run.json").open("r", encoding="utf-8") as stream:
                runs.append(json.load(stream))
        runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return runs

    def best_run(self, metric: str, mode: str = "max") -> tuple[dict[str, Any], float] | None:
        best: tuple[dict[str, Any], float] | None = None
        for run in self.list_runs():
            for entry in run.get("eval_history", []):
                if metric not in entry:
                    continue
                value = entry[metric]
                if best is None or (mode == "max") == (value > best[1]):
                    best = (run, value)
        return best

    def lineage(self, run_id: str) -> list[dict[str, Any]]:
        """Return the resume chain from the root ancestor down to ``run_id``."""
        chain: list[dict[str, Any]] = []
        current: str | None = run_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            try:
                run = self.get(current)
            except FileNotFoundError:
                break
            chain.append(run)
            current = run.get("parent_run_id")
        return list(reversed(chain))

    def resolve_checkpoint(self, run_id: str, which: int | str = "latest") -> Path:
        """Path to a checkpoint. ``which`` is "latest" (default), "best", or an epoch."""
        run = self.get(run_id)
        ckpt_dir = self.root / run_id / "checkpoints"
        if which == "best":
            best = run.get("best_checkpoint")
            return ckpt_dir / (best["file"] if best else "best.pth")
        if which == "latest":
            checkpoints = run.get("checkpoints", [])
            if checkpoints:
                latest = max(checkpoints, key=lambda c: c["epoch"])
                return ckpt_dir / latest["file"]
            return ckpt_dir / "best.pth"  # fallback if nothing recorded
        return ckpt_dir / f"epoch{int(which)}.pth"


# --------------------------------------------------------------------------- CLI
def _fmt_eval(run: dict[str, Any]) -> str:
    history = run.get("eval_history", [])
    if not history:
        return "-"
    last = history[-1]
    parts = [f"{k}={v:.1f}" for k, v in last.items() if k != "epoch"]
    return " ".join(parts) if parts else "-"


def _cmd_list(reg: Registry, _: argparse.Namespace) -> None:
    runs = reg.list_runs()
    if not runs:
        print(f"No runs found under {reg.root}/")
        return
    print(f"{'run_id':<28} {'status':<10} {'epochs':>6}  {'title':<24} latest eval")
    print("-" * 100)
    for run in runs:
        title = (run.get("title") or "")[:24]
        print(
            f"{run['run_id']:<28} {run.get('status', '?'):<10} "
            f"{run.get('epochs_completed', 0):>6}  {title:<24} {_fmt_eval(run)}"
        )


def _cmd_show(reg: Registry, args: argparse.Namespace) -> None:
    run = reg.get(args.run_id)
    print(json.dumps(run, indent=2, ensure_ascii=False))


def _cmd_lineage(reg: Registry, args: argparse.Namespace) -> None:
    chain = reg.lineage(args.run_id)
    if not chain:
        print(f"No run '{args.run_id}'.")
        return
    for depth, run in enumerate(chain):
        prefix = "   " * depth + ("└─ " if depth else "")
        print(
            f"{prefix}{run['run_id']}  (epochs={run.get('epochs_completed', 0)})  {_fmt_eval(run)}"
        )


def _cmd_graph(reg: Registry, args: argparse.Namespace) -> None:
    from kaisparov.tracking.lineage_view import build_html

    html = build_html(reg.list_runs(), root=str(reg.root))
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote lineage graph to {out}")
    if not args.no_open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())


def _cmd_best(reg: Registry, args: argparse.Namespace) -> None:
    result = reg.best_run(args.metric, mode=args.mode)
    if result is None:
        print(f"No run has eval metric '{args.metric}'.")
        return
    run, value = result
    ckpt = reg.resolve_checkpoint(run["run_id"], "best")
    print(f"Best by {args.metric} ({args.mode}): {value:.2f}")
    print(f"  run_id     : {run['run_id']}")
    print(f"  checkpoint : {ckpt}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kaisparov runs", description="Inspect training runs.")
    parser.add_argument("--runs-dir", default="runs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all runs")

    show = sub.add_parser("show", help="Print a run's full metadata")
    show.add_argument("run_id")

    lineage = sub.add_parser("lineage", help="Show a run's resume chain")
    lineage.add_argument("run_id")

    best = sub.add_parser("best", help="Find the best run by an eval metric")
    best.add_argument("--metric", default="elo_vs_random")
    best.add_argument("--mode", default="max", choices=["max", "min"])

    graph = sub.add_parser("graph", help="Render the lineage as a git-log-style HTML page")
    graph.add_argument("-o", "--output", default="runs/lineage.html", help="Output HTML file")
    graph.add_argument("--no-open", action="store_true", help="Do not open a browser")

    args = parser.parse_args(argv)
    reg = Registry(args.runs_dir)
    commands = {
        "list": _cmd_list,
        "show": _cmd_show,
        "lineage": _cmd_lineage,
        "best": _cmd_best,
        "graph": _cmd_graph,
    }
    commands[args.command](reg, args)


if __name__ == "__main__":
    main()
