"""Tests for the opponent-pool preset system (config/pools.yaml) and its parsing."""

from __future__ import annotations

import pytest

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.training.config import PoolSpec, TrainConfig, build_pool_spec
from kaisparov.training.trainer import _build_pool_baseline


def test_build_pool_spec_from_inline_mapping():
    spec = build_pool_spec(
        {
            "opponents": [
                {"kind": "material", "weight": 3.0, "params": {"avoid_king_suicide": True}},
                {"kind": "random", "weight": 1.0},
                {"kind": "snapshot", "count": 6, "every": 15, "params": {"depth": 2}},
            ],
            "group_weights": {"baseline": 1.5, "snapshot": 1.0},
        }
    )
    assert isinstance(spec, PoolSpec)
    assert [o.kind for o in spec.baselines] == ["material", "random"]
    assert spec.baselines[0].weight == 3.0
    assert spec.baselines[0].params["avoid_king_suicide"] is True
    # kind=snapshot defaults its group to "snapshot".
    snap = spec.snapshot
    assert snap is not None
    assert snap.count == 6 and snap.every == 15 and snap.params["depth"] == 2
    assert spec.group_weights == {"baseline": 1.5, "snapshot": 1.0}


def test_build_pool_spec_from_named_preset():
    # Presets shipped in config/pools.yaml.
    spec = build_pool_spec("sound_v4")
    kinds = {o.kind for o in spec.opponents}
    assert {"material", "random", "snapshot"} <= kinds
    assert spec.snapshot.params["avoid_king_suicide"] is True


def test_unknown_preset_and_kind_raise():
    with pytest.raises(ValueError, match="Unknown pool preset"):
        build_pool_spec("does_not_exist")
    with pytest.raises(ValueError, match="Unknown opponent kind"):
        build_pool_spec({"opponents": [{"kind": "wizard"}]})


def test_at_most_one_snapshot_entry():
    with pytest.raises(ValueError, match="at most one 'snapshot'"):
        build_pool_spec(
            {"opponents": [{"kind": "snapshot"}, {"kind": "snapshot", "params": {"depth": 2}}]}
        )


def test_config_expands_pool_preset_into_persisted_config():
    c = TrainConfig.from_dict({"rollout": {"opponent": "pool", "pool": "teachers_v3"}})
    # The preset name is resolved to an inline mapping (self-contained, reproducible).
    assert isinstance(c.rollout.pool, dict)
    assert c.rollout.pool["preset"] == "teachers_v3"
    assert "opponents" in c.rollout.pool
    # And it round-trips through to_dict/from_dict unchanged.
    again = TrainConfig.from_dict(c.to_dict())
    assert again.rollout.pool == c.rollout.pool


def test_config_rejects_bad_preset_eagerly():
    with pytest.raises(ValueError, match="Unknown pool preset"):
        TrainConfig.from_dict({"rollout": {"pool": "nope"}})


def test_build_pool_baseline_model_free_kinds_use_params():
    from kaisparov.training.config import OpponentSpec

    mat = _build_pool_baseline(
        None, None, 8, OpponentSpec(kind="material", params={"avoid_king_suicide": True}), seed=7
    )
    rnd = _build_pool_baseline(None, None, 8, OpponentSpec(kind="random"), seed=7)
    assert isinstance(mat, MaterialAgent) and mat.avoid_king_suicide is True
    assert isinstance(rnd, RandomAgent) and rnd.avoid_king_suicide is False


def test_build_pool_baseline_neural_requires_checkpoint():
    from kaisparov.training.config import OpponentSpec

    with pytest.raises(ValueError, match="needs params.checkpoint"):
        _build_pool_baseline(None, None, 8, OpponentSpec(kind="minimax"), seed=0)


def test_trainer_builds_pool_from_preset(tmp_path):
    from kaisparov.training.trainer import Trainer

    cfg = TrainConfig.from_dict(
        {
            "hidden_dim": 8,
            "runs_dir": str(tmp_path),
            "rollout": {"opponent": "pool", "pool": "sound_v4"},
        }
    )
    trainer = Trainer(cfg)
    assert trainer.pool is not None
    assert len(trainer.pool) >= 2  # material + random baselines seeded from epoch 1
    assert trainer.take_snapshots is True
    assert trainer.snapshot_every == 15  # from sound_v4's snapshot.every


def test_trainer_baseline_only_pool_disables_snapshots(tmp_path):
    from kaisparov.training.trainer import Trainer

    cfg = TrainConfig.from_dict(
        {
            "hidden_dim": 8,
            "runs_dir": str(tmp_path),
            "rollout": {"opponent": "pool", "pool": {"opponents": [{"kind": "material"}]}},
        }
    )
    trainer = Trainer(cfg)
    assert trainer.take_snapshots is False  # no snapshot entry -> pool never grows
    assert len(trainer.pool) == 1
