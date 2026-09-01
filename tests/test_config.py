"""Tests for the typed training config."""

from __future__ import annotations

from kaisparov.training.config import TrainConfig


def test_defaults():
    c = TrainConfig()
    assert c.model == "gnn_v1"
    assert c.ppo.gamma == 0.99
    assert c.ppo.self_play is True
    assert c.eval.every == 5


def test_from_dict_nested_and_ignores_unknown():
    c = TrainConfig.from_dict(
        {
            "model": "gnn_v1",
            "epochs": 3,
            "ppo": {"gamma": 0.5, "bogus": 123},
            "unknown_top": 9,
        }
    )
    assert c.epochs == 3
    assert c.ppo.gamma == 0.5  # nested override applied
    assert c.ppo.clip_eps == 0.2  # untouched default kept
    assert not hasattr(c, "unknown_top")


def test_yaml_roundtrip(tmp_path):
    c = TrainConfig(seed=7)
    c.ppo.entropy_coef = 0.05
    path = tmp_path / "c.yaml"
    c.to_yaml(path)
    reloaded = TrainConfig.from_yaml(path)
    assert reloaded.to_dict() == c.to_dict()
