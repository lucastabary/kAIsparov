"""Training entry point: config-driven PPO self-play.

Usage:
    python -m kaisparov.train --config config/default.yaml
    python -m kaisparov.train --epochs 100 --hidden-dim 16 --cpu
"""

from __future__ import annotations

import argparse

from kaisparov.training.config import TrainConfig
from kaisparov.training.trainer import Trainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="kaisparov train", description="PPO self-play training.")
    parser.add_argument("--config", default=None, help="Path to a YAML config file.")
    parser.add_argument(
        "--resume", default=None, help="Continue from an existing run id (inherits its config)."
    )
    parser.add_argument("--runs-dir", default="runs", help="Where runs live (for --resume).")
    # Optional overrides (applied on top of the config / defaults).
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Self-play games per epoch.")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args(argv)


def _resume_config(run_id: str, runs_dir: str) -> TrainConfig:
    """Rebuild a config from an existing run and point it at that run's checkpoint."""
    from kaisparov.tracking.registry import Registry

    reg = Registry(runs_dir)
    parent = reg.get(run_id)
    config = TrainConfig.from_dict(parent.get("config", {}))
    config.parent_run_id = run_id
    config.runs_dir = runs_dir

    checkpoint = reg.resolve_checkpoint(run_id, "latest")  # continue from where it stopped
    if not checkpoint.exists():
        raise SystemExit(f"Run '{run_id}' has no checkpoint to resume from.")
    config.resume_from = str(checkpoint)
    return config


def build_config(args: argparse.Namespace) -> TrainConfig:
    if args.resume is not None:
        config = _resume_config(args.resume, args.runs_dir)
    elif args.config:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig()

    if args.model is not None:
        config.model = args.model
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.seed is not None:
        config.seed = args.seed
    if args.episodes is not None:
        config.rollout.episodes_per_epoch = args.episodes
    if args.notes is not None:
        config.notes = args.notes
    if args.cpu:
        config.device = "cpu"
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    Trainer(config).train()


if __name__ == "__main__":
    main()
