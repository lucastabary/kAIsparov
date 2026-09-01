"""Typed, YAML-backed training configuration.

Every knob lives here with a sensible default, so a run is fully described by one
config object (persisted alongside its checkpoints for reproducibility).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PPOSettings:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    self_play: bool = True


@dataclass
class RolloutSettings:
    episodes_per_epoch: int = 8
    max_steps_per_episode: int = 100


@dataclass
class CurriculumSettings:
    name: str = "Phase 1: 6 pieces"
    max_pieces_per_side: int = 6
    allow_major: bool = False
    allow_minor: bool = True
    allow_pawns: bool = True


@dataclass
class EvalSettings:
    enabled: bool = True
    every: int = 5  # epochs
    games: int = 20
    max_plies: int = 200


@dataclass
class TrainConfig:
    model: str = "gnn_v1"
    hidden_dim: int = 8
    epochs: int = 50
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    checkpoint_every: int = 10
    runs_dir: str = "runs"
    notes: str = ""

    # Lineage: set by `--resume` to continue from an earlier run's checkpoint.
    resume_from: str | None = None
    parent_run_id: str | None = None

    ppo: PPOSettings = field(default_factory=PPOSettings)
    rollout: RolloutSettings = field(default_factory=RolloutSettings)
    curriculum: CurriculumSettings = field(default_factory=CurriculumSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)

    # --------------------------------------------------------------- (de)serialize
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        nested = {
            "ppo": PPOSettings,
            "rollout": RolloutSettings,
            "curriculum": CurriculumSettings,
            "eval": EvalSettings,
        }
        data = dict(data or {})
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue  # ignore unknown top-level keys
            if key in nested and isinstance(value, dict):
                sub_cls = nested[key]
                sub_known = {f.name for f in fields(sub_cls)}
                kwargs[key] = sub_cls(**{k: v for k, v in value.items() if k in sub_known})
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(yaml.safe_load(stream) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)
