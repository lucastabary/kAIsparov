"""Training entry point: config-driven PPO self-play.

Usage:
    python -m kaisparov.train --config config/default.yaml
    python -m kaisparov.train --epochs 100 --hidden-dim 16 --cpu
"""

from __future__ import annotations

import argparse

import yaml

from kaisparov.training.config import TrainConfig, build_resume_config, load_train_config
from kaisparov.training.trainer import Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kaisparov train", description="PPO self-play training.")
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Continue from a run id (overrides config's resume_from_run).",
    )
    parser.add_argument("--runs-dir", default="runs", help="Where runs live (for --resume).")
    # Optional overrides (applied on top of the config / defaults).
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Self-play games per epoch.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TrainConfig:
    raw: dict = {}
    if args.config:
        with open(args.config, encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

    resume_run = args.resume or raw.get("resume_from_run")
    if resume_run:
        # Inherit architecture (and anything not overridden) from the parent run.
        config = build_resume_config(resume_run, raw, args.runs_dir)
    elif args.config:
        config = load_train_config(args.config, args.runs_dir)
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
    config = build_config(args)
    Trainer(config).train()


if __name__ == "__main__":
    main()
