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
    value_coef: float = 0.25  # keep the critic term from dominating the shared trunk
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    self_play: bool = True


@dataclass
class RolloutSettings:
    episodes_per_epoch: int = 8
    max_steps_per_episode: int = 100
    opponent: str = "self"  # "self" (self-play) | "pool" (league vs past checkpoints)
    pool_size: int = 5  # how many past snapshots to keep
    snapshot_every: int = 20  # add the learner to the pool every N epochs (pool mode)
    # Fixed baseline opponents seeded into the pool from epoch 1 (pool mode only),
    # e.g. ["material", "random"] — an opponent curriculum that punishes hung pieces
    # and king exposure long before the first self-snapshot exists.
    baselines: list[str] = field(default_factory=list)
    # Weighted sampling (pool mode). By default the pool draws uniformly over
    # baselines + snapshots, so accumulating snapshots dilute the fixed baselines
    # (Material's share falls from 1/2 of games to ~1/7 as the pool fills). Set
    # `baseline_weight`/`snapshot_weight` to give the two GROUPS a fixed relative
    # share instead — the tactical teachers keep their weight no matter how many
    # snapshots exist. Leave both None for the legacy uniform draw.
    baseline_weight: float | None = None
    snapshot_weight: float | None = None
    # Relative weights of the individual baselines, same order as `baselines`
    # (e.g. [3, 1] = Material drawn 3x as often as Random). Empty = uniform.
    baseline_weights: list[float] = field(default_factory=list)
    # Wrap frozen snapshots in a Minimax alpha-beta search of this depth, so past
    # selves refute one-move blunders instead of just sampling their policy (0 = raw
    # policy reply, the default; 1 = cheap 1-ply lookahead that still catches every
    # king capture; 2 = stronger but ~b x costlier). Costly on CPU with many pieces.
    snapshot_search_depth: int = 0


@dataclass
class CurriculumSettings:
    name: str = "Phase 1: 6 pieces"
    max_pieces_per_side: int = 6
    allow_major: bool = False
    allow_minor: bool = True
    allow_pawns: bool = True
    # Guarantee neither king starts on an attacked square (no ply-0 free capture).
    ensure_kings_safe: bool = True


@dataclass
class EvalSettings:
    enabled: bool = True
    every: int = 5  # epochs
    games: int = 20
    max_plies: int = 200


@dataclass
class RewardSettings:
    """Weighted reward-shaping terms (from the mover's point of view, per ply)."""

    preset: str = ""  # name if resolved from config/rewards.yaml (for the record)
    material: float = 1.0  # * value of the captured (non-king) piece
    king_capture: float = 0.0  # flat reward for capturing the king (winning the game)
    check: float = 0.0  # bonus if the move leaves the opponent in check
    king_safety: float = 0.0  # penalty per ply your move leaves your own king capturable
    step_penalty: float = 0.0  # subtracted every ply (rewards decisive play)


def _load_reward_presets() -> dict[str, Any]:
    for candidate in (
        Path("config/rewards.yaml"),
        Path(__file__).resolve().parents[3] / "config" / "rewards.yaml",
    ):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as stream:
                return yaml.safe_load(stream) or {}
    return {}


def _build_reward(value: Any) -> RewardSettings:
    """Build RewardSettings from a preset name (str) or an inline mapping (dict)."""
    known = {f.name for f in fields(RewardSettings)}
    if isinstance(value, str):
        presets = _load_reward_presets()
        if value not in presets:
            raise ValueError(
                f"Unknown reward preset '{value}'. Available: {sorted(presets)} "
                "(define them in config/rewards.yaml)."
            )
        data = {"preset": value, **(presets[value] or {})}
    elif isinstance(value, dict):
        data = dict(value)
    else:
        return RewardSettings()
    return RewardSettings(**{k: v for k, v in data.items() if k in known})


@dataclass
class TrainConfig:
    model: str = "rgcn"
    hidden_dim: int = 8
    epochs: int = 50
    seed: int = 0
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    checkpoint_every: int = 10
    runs_dir: str = "runs"

    # Free-text documentation of this run's intent (shown by `kaisparov runs`).
    title: str = ""
    description: str = ""
    notes: str = ""

    # Lineage. Set `resume_from_run` (a run id) in the YAML to continue an earlier
    # run: its architecture (and any field you don't override) is inherited.
    resume_from_run: str | None = None
    resume_from: str | None = None  # resolved checkpoint path (filled automatically)
    parent_run_id: str | None = None

    ppo: PPOSettings = field(default_factory=PPOSettings)
    rollout: RolloutSettings = field(default_factory=RolloutSettings)
    # None = train from the normal starting position (no curriculum).
    curriculum: CurriculumSettings | None = None
    eval: EvalSettings = field(default_factory=EvalSettings)
    reward: RewardSettings = field(default_factory=RewardSettings)

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
            if key == "reward":  # str preset or inline mapping
                kwargs[key] = _build_reward(value)
            elif key in nested and isinstance(value, dict):
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


# Fields inferred from the parent run on resume (must match the checkpoint), so you
# never redefine the architecture when continuing a run.
_INHERITED_ARCHITECTURE = ("model", "hidden_dim")


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``over`` on ``base`` (one level of nested dicts)."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_resume_config(
    run_id: str, overrides: dict[str, Any] | None = None, runs_dir: str = "runs"
) -> TrainConfig:
    """Config for a run that continues ``run_id``.

    The parent's config is the base; ``overrides`` (e.g. the new YAML) win on top.
    The architecture is always taken from the parent, the parent's latest checkpoint
    is resolved, and lineage/doc fields are set for the *new* run.
    """
    from kaisparov.tracking.registry import Registry

    reg = Registry(runs_dir)
    parent = reg.get(run_id)
    parent_cfg: dict[str, Any] = parent.get("config", {})
    overrides = overrides or {}

    config = TrainConfig.from_dict(_deep_merge(parent_cfg, overrides))
    for name in _INHERITED_ARCHITECTURE:
        setattr(config, name, parent_cfg.get(name, getattr(config, name)))

    config.runs_dir = runs_dir
    config.parent_run_id = run_id
    config.resume_from_run = run_id
    checkpoint = reg.resolve_checkpoint(run_id, "latest")
    if not checkpoint.exists():
        raise SystemExit(f"Run '{run_id}' has no checkpoint to resume from.")
    config.resume_from = str(checkpoint)

    # Documentation describes THIS run, not the parent's — don't inherit it.
    config.title = overrides.get("title", "")
    config.description = overrides.get("description", "")
    return config


def load_train_config(path: str | Path | None, runs_dir: str = "runs") -> TrainConfig:
    """Load a training config, resolving ``resume_from_run`` if present."""
    if path is None:
        return TrainConfig()
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    run_id = raw.get("resume_from_run")
    if run_id:
        return build_resume_config(run_id, raw, raw.get("runs_dir", runs_dir))
    return TrainConfig.from_dict(raw)
