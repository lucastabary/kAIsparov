"""``kaisparov bench`` — generate problem suites and run models on them.

    kaisparov bench generators                         # list the problem families
    kaisparov bench generate config/benchmarks/smoke.yaml -o data/bench/smoke.jsonl
    kaisparov bench run config/benchmarks/smoke.yaml -a random -a material \\
        -a run:20260903-155710_rgcn@best -a "v2+search=run:<id>@best+minimax2"
    kaisparov bench show runs/benchmarks/<report>.json

``run`` accepts a spec (``.yaml``, built on the fly) or a frozen suite (``.jsonl``),
and writes its report under ``runs/benchmarks/`` unless ``--out`` says otherwise —
next to the training runs, git-ignored like them, and invisible to ``kaisparov runs``
(no ``run.json``).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from kaisparov.bench.contestants import Contestant
from kaisparov.bench.generators.base import ProblemGenerator
from kaisparov.bench.problem import Problem
from kaisparov.bench.report import BenchmarkReport
from kaisparov.bench.runner import BenchmarkRunner
from kaisparov.bench.suite import Suite
from kaisparov.bench.tasks import Outcome


def _cmd_generators(_: argparse.Namespace) -> None:
    for name, generator in ProblemGenerator.available().items():
        print(f"{name:<16} [{generator.theme}] {generator.description}")


def _describe(suite: Suite) -> str:
    themes = Counter(problem.theme for problem in suite)
    parts = ", ".join(f"{theme}={n}" for theme, n in sorted(themes.items()))
    return f"{suite.name}: {len(suite)} problems ({parts})"


def _cmd_generate(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    suite = Suite.load(args.suite)
    print(f"{_describe(suite)} in {time.perf_counter() - start:.1f}s")
    if args.output:
        print(f"Wrote {suite.save(args.output)}")


def _progress(contestant: Contestant, problem: Problem, outcome: Outcome, i: int, n: int) -> None:
    solved = "ok" if outcome.solved else "--"
    sys.stdout.write(f"\r  {contestant.name}: {i}/{n} [{solved}] {problem.id:<28}")
    if i == n:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _cmd_run(args: argparse.Namespace) -> None:
    if not args.agent:
        raise SystemExit("Name at least one contestant with -a (e.g. -a material).")
    suite = Suite.load(args.suite).select(themes=args.themes, limit=args.limit)
    if not len(suite):
        raise SystemExit("No problem left after filtering.")
    contestants = [Contestant.parse(spec, runs_dir=args.runs_dir) for spec in args.agent]
    print(_describe(suite))

    runner = BenchmarkRunner(suite, seed=args.seed, progress=None if args.quiet else _progress)
    report = runner.run(contestants)
    report.meta["source"] = str(args.suite)

    out = args.out
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = Path(args.runs_dir) / "benchmarks" / f"{stamp}_{suite.name}.json"
    print()
    print(report.format_table())
    print(f"\nWrote {report.save(out)}")


def _cmd_show(args: argparse.Namespace) -> None:
    report = BenchmarkReport.merge(BenchmarkReport.load(path) for path in args.reports)
    print(f"{report.suite} ({len(args.reports)} report(s))\n")
    print(report.format_table(metric=args.metric, intervals=not args.no_intervals))
    if args.output:
        print(f"\nWrote {report.save(args.output)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kaisparov bench", description="Benchmark models.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generators", help="List the registered problem generators")

    generate = sub.add_parser("generate", help="Build a suite from a YAML spec")
    generate.add_argument("suite", help="Suite spec (.yaml) or frozen suite (.jsonl)")
    generate.add_argument("-o", "--output", help="Freeze the problems to this .jsonl file")

    run = sub.add_parser("run", help="Run contestants on a suite")
    run.add_argument("suite", help="Suite spec (.yaml) or frozen suite (.jsonl)")
    run.add_argument(
        "-a", "--agent", action="append", help="Contestant spec, repeatable (see module doc)"
    )
    run.add_argument("--themes", nargs="+", help="Only these themes")
    run.add_argument("--limit", type=int, help="At most N problems per theme")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--runs-dir", default="runs")
    run.add_argument("--out", help="Report path (default: runs/benchmarks/<stamp>_<suite>.json)")
    run.add_argument("-q", "--quiet", action="store_true", help="No per-problem progress")

    show = sub.add_parser("show", help="Print saved reports (several are merged)")
    show.add_argument("reports", nargs="+")
    show.add_argument("--metric", choices=["solve_rate", "score"], default="solve_rate")
    show.add_argument("--no-intervals", action="store_true")
    show.add_argument("-o", "--output", help="Save the merged report to this path")

    args = parser.parse_args(argv)
    commands = {
        "generators": _cmd_generators,
        "generate": _cmd_generate,
        "run": _cmd_run,
        "show": _cmd_show,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
