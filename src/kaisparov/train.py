"""Training entry point: config-driven PPO self-play.

Usage:
    python -m kaisparov.train --config config/default.yaml
    python -m kaisparov.train --epochs 100 --hidden-dim 16 --cpu

Chaining (curriculum in one command): pass several configs to ``--config`` and
each stage after the first resumes from the run the previous stage produced (its
latest checkpoint), so a whole recipe runs end to end without hand-copying run ids::

    python -m kaisparov.train --config \
        config/experiments/scratch_stage1.yaml \
        config/experiments/scratch_stage2.yaml \
        config/experiments/scratch_stage3.yaml
"""

from __future__ import annotations

import argparse

import yaml

from kaisparov.training.config import TrainConfig, build_resume_config, load_train_config
from kaisparov.training.trainer import Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kaisparov train", description="PPO self-play training.")
    parser.add_argument(
        "--config",
        default=None,
        nargs="+",
        metavar="YAML",
        help=(
            "Path to a YAML config file. Pass several to chain a curriculum: each "
            "stage after the first resumes from the run the previous stage produced "
            "(its latest checkpoint), so any `resume_from_run` in those YAMLs is "
            "overridden by the actual parent run id."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Continue from a run id (overrides config's resume_from_run).",
    )
    parser.add_argument("--runs-dir", default="runs", help="Where runs live (for --resume).")
    # Optional overrides (applied on top of the config / defaults).
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Save a checkpoint every N epochs (applied to every chained stage; "
        "set it <= --epochs so a short run still leaves a checkpoint to resume from).",
    )
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Self-play games per epoch.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args(argv)


def build_config(
    args: argparse.Namespace,
    config_path: str | None = None,
    resume_run_id: str | None = None,
) -> TrainConfig:
    """Build the config for a single training stage.

    ``config_path`` is the YAML for this stage (``None`` -> defaults).
    ``resume_run_id`` is the run the previous stage produced when chaining; it
    overrides both ``--resume`` and the YAML's ``resume_from_run`` so the stage
    continues from that run's latest checkpoint.
    """
    raw: dict = {}
    if config_path:
        with open(config_path, encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

    resume_run = resume_run_id or args.resume or raw.get("resume_from_run")
    if resume_run:
        # Inherit architecture (and anything not overridden) from the parent run.
        config = build_resume_config(resume_run, raw, args.runs_dir)
    elif config_path:
        config = load_train_config(config_path, args.runs_dir)
    else:
        config = TrainConfig()

    is_resume = resume_run is not None
    if not is_resume:  # architecture is locked when resuming
        if args.model is not None:
            config.model = args.model
        if args.hidden_dim is not None:
            config.hidden_dim = args.hidden_dim
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.checkpoint_every is not None:
        config.checkpoint_every = args.checkpoint_every
    if args.seed is not None:
        config.seed = args.seed
    if args.episodes is not None:
        config.rollout.episodes_per_epoch = args.episodes
    if args.title is not None:
        config.title = args.title
    if args.description is not None:
        config.description = args.description
    if args.cpu:
        config.device = "cpu"
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # `--config` may name several stages: run them in sequence, each resuming from
    # the run the previous one produced. `[None]` = no config -> a single default run.
    stages: list[str | None] = list(args.config) if args.config else [None]

    prev_run_id: str | None = None
    completed: list[str] = []
    for index, config_path in enumerate(stages):
        if len(stages) > 1:
            label = config_path or "default"
            print(f"\n=== stage {index + 1}/{len(stages)}: {label} ===")
            if prev_run_id is not None:
                print(f"    resuming from run {prev_run_id}")
        # Only stages after the first chain onto the previous run; the first stage
        # still honours an explicit --resume / resume_from_run of its own.
        resume_run_id = prev_run_id if index > 0 else None
        config = build_config(args, config_path, resume_run_id)
        prev_run_id = Trainer(config).train()
        completed.append(prev_run_id)

    if len(completed) > 1:
        print(f"\nChain complete. Lineage: {' -> '.join(completed)}")


if __name__ == "__main__":
    main()
