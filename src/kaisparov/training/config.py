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
    # Collection is CPU-bound in the pure-Python engine; split the epoch's episodes
    # across this many worker processes (each with its own model copy) and merge their
    # buffers. 1 = in-process (default). 0 = auto (os.cpu_count()). Both self-play and
    # pool/league collection are parallelised (workers rebuild the pool from the spec).
    num_workers: int = 1
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
    # Make every pool opponent (baselines + snapshots) refuse moves that hang their
    # own king (one-ply king safety guard, see kaisparov.agents.safety). Off by
    # default. Turning it on removes the "rush the enemy king" free win from
    # self-play — the opponent no longer leaves its king en prise — so the learner
    # must win soundly instead of racing. May slow early training (a blind rush
    # stops working before the model has learned anything else); set per config to
    # test. Does NOT affect the eval baselines, which stay canonical.
    opponent_avoid_king_suicide: bool = False
    # Fine-grained opponent pool. Either a preset name from config/pools.yaml, or an
    # inline mapping with the same shape (a flat ``opponents`` list, each entry with
    # its own kind / group / weight / count / per-agent ``params``, plus optional
    # ``group_weights``). When set it fully defines the pool and OVERRIDES the flat
    # legacy fields above (baselines / *_weight / pool_size / snapshot_every /
    # snapshot_search_depth / opponent_avoid_king_suicide). Leave None to use them.
    # See PoolSpec / config/pools.yaml. Resolved (expanded) at load time so the run's
    # persisted config records the actual opponents, not just a preset name.
    pool: Any = None


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


# ------------------------------------------------------------------- opponent pool
# Opponent kinds usable in a pool preset. ``random``/``material`` need no model;
# ``neural``/``minimax`` load a frozen model from a checkpoint (``params.checkpoint``);
# ``snapshot`` is the stream of frozen past-selves of the learner, added over time.
KNOWN_OPPONENT_KINDS = frozenset({"random", "material", "neural", "minimax", "snapshot"})


@dataclass
class OpponentSpec:
    """One entry of a pool preset (see :class:`PoolSpec`).

    ``group`` is either ``"baseline"`` (a fixed opponent present from epoch 1) or
    ``"snapshot"`` (the accumulating stream of frozen past-selves). ``params`` holds
    that agent's own hyper-parameters, e.g. ``seed``, ``avoid_king_suicide`` for the
    baselines; ``depth`` (0 = raw policy, >=1 = minimax lookahead), ``deterministic``,
    ``avoid_king_suicide`` for snapshots; ``checkpoint``/``depth`` for a frozen
    neural/minimax baseline loaded from a past run.
    """

    kind: str
    group: str = "baseline"
    weight: float = 1.0
    count: int = 1  # snapshot group: max past-selves retained in the pool
    every: int = 20  # snapshot group: add one frozen self every N epochs
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolSpec:
    """A fully-resolved opponent pool: a flat list of opponents + group weights."""

    opponents: list[OpponentSpec] = field(default_factory=list)
    group_weights: dict[str, float] = field(default_factory=dict)

    @property
    def baselines(self) -> list[OpponentSpec]:
        return [o for o in self.opponents if o.group == "baseline"]

    @property
    def snapshot(self) -> OpponentSpec | None:
        snaps = [o for o in self.opponents if o.group == "snapshot"]
        if len(snaps) > 1:
            raise ValueError("A pool preset supports at most one 'snapshot' opponent entry.")
        return snaps[0] if snaps else None


def _load_pool_presets() -> dict[str, Any]:
    for candidate in (
        Path("config/pools.yaml"),
        Path(__file__).resolve().parents[3] / "config" / "pools.yaml",
    ):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as stream:
                return yaml.safe_load(stream) or {}
    return {}


def _expand_pool(value: Any) -> Any:
    """Resolve a preset name (str) to its inline mapping; pass a mapping through.

    Keeps the ``preset`` name in the expanded mapping (for the record), mirroring how
    reward presets are stored, so the persisted run config is self-contained.
    """
    if isinstance(value, str):
        presets = _load_pool_presets()
        if value not in presets:
            raise ValueError(
                f"Unknown pool preset '{value}'. Available: {sorted(presets)} "
                "(define them in config/pools.yaml)."
            )
        return {"preset": value, **(presets[value] or {})}
    if isinstance(value, dict):
        return dict(value)
    return value


def build_pool_spec(value: Any) -> PoolSpec:
    """Build a :class:`PoolSpec` from a preset name (str) or an inline mapping."""
    data = _expand_pool(value)
    if not isinstance(data, dict):
        return PoolSpec()
    opponents: list[OpponentSpec] = []
    for entry in data.get("opponents", []):
        item = dict(entry)
        kind = item.get("kind")
        if kind not in KNOWN_OPPONENT_KINDS:
            raise ValueError(
                f"Unknown opponent kind {kind!r}. Expected one of {sorted(KNOWN_OPPONENT_KINDS)}."
            )
        group = item.get("group", "snapshot" if kind == "snapshot" else "baseline")
        if group not in ("baseline", "snapshot"):
            raise ValueError(f"Opponent group must be 'baseline' or 'snapshot', got {group!r}.")
        opponents.append(
            OpponentSpec(
                kind=kind,
                group=group,
                weight=float(item.get("weight", 1.0)),
                count=int(item.get("count", 1)),
                every=int(item.get("every", 20)),
                params=dict(item.get("params", {})),
            )
        )
    spec = PoolSpec(opponents=opponents, group_weights=dict(data.get("group_weights", {})))
    _ = spec.snapshot  # validate eagerly: raises if more than one snapshot entry
    return spec


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
                sub_kwargs = {k: v for k, v in value.items() if k in sub_known}
                # Expand a pool preset name to its inline mapping so the persisted
                # config records the actual opponents (validated eagerly).
                if key == "rollout" and sub_kwargs.get("pool") is not None:
                    sub_kwargs["pool"] = _expand_pool(sub_kwargs["pool"])
                    build_pool_spec(sub_kwargs["pool"])  # fail fast on a bad preset
                kwargs[key] = sub_cls(**sub_kwargs)
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
