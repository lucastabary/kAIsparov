"""Tests for the entropy thermostat (ppo.target_entropy): the coefficient adapts to hold
the policy's normalised entropy at a target, is logged, and survives a resume."""

from __future__ import annotations

import json

import pytest
import torch

from kaisparov.core.game import ChessGame
from kaisparov.models.factory import load_backend, load_backend_spec
from kaisparov.training.config import TrainConfig
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum
from kaisparov.training.ppo import adapt_entropy_coef


def test_the_coefficient_goes_up_below_the_target_and_down_above():
    below = adapt_entropy_coef(0.01, entropy_norm=0.1, target=0.3, lr=0.5, low=1e-3, high=0.05)
    above = adapt_entropy_coef(0.01, entropy_norm=0.8, target=0.3, lr=0.5, low=1e-3, high=0.05)
    on_target = adapt_entropy_coef(0.01, entropy_norm=0.3, target=0.3, lr=0.5, low=1e-3, high=0.05)
    assert below > 0.01 > above
    assert on_target == pytest.approx(0.01)
    # Proportional: a gap of 0.1 at lr 0.5 is ~5%.
    step = adapt_entropy_coef(0.01, entropy_norm=0.2, target=0.3, lr=0.5, low=1e-3, high=0.05)
    assert step == pytest.approx(0.01 * 1.0513, rel=1e-3)


def test_the_coefficient_stays_within_its_bounds():
    coef = 0.01
    for _ in range(200):  # a policy stuck far below the target
        coef = adapt_entropy_coef(coef, entropy_norm=0.0, target=0.5, lr=0.5, low=1e-3, high=0.05)
    assert coef == 0.05
    for _ in range(200):  # ... and far above it
        coef = adapt_entropy_coef(coef, entropy_norm=1.0, target=0.1, lr=0.5, low=1e-3, high=0.05)
    assert coef == 1e-3


def test_ppo_reports_the_normalised_entropy():
    device = torch.device("cpu")
    spec = load_backend_spec("rgcn")
    agent = spec.model_class.create_agent(device=device, hidden_dim=8)
    optimizer = spec.model_class.create_optimizer(agent, learning_rate=1e-3)
    buffer = spec.buffer_class()
    curriculum = PieceCountCurriculum(PhaseConfig(name="t", max_pieces_per_side=4), seed=0)
    spec.collect_data(
        agent,
        ChessGame(),
        buffer,
        num_episodes=2,
        max_steps_per_episode=12,
        model_module=load_backend("rgcn"),
        curriculum=curriculum,
    )
    metrics = spec.train_one_epoch(agent, buffer, optimizer, device=device, update_epochs=1)
    # An untrained policy is close to uniform over the legal moves.
    assert 0.5 < metrics["entropy_norm"] <= 1.0


def _config(tmp_path, **ppo) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "hidden_dim": 8,
            "epochs": 2,
            "checkpoint_every": 1,
            "runs_dir": str(tmp_path),
            "rollout": {"episodes_per_epoch": 2, "max_steps_per_episode": 12},
            "curriculum": {"max_pieces_per_side": 3},
            "eval": {"enabled": False},
            "ppo": {"update_epochs": 1, "entropy_coef": 0.01, **ppo},
        }
    )


def _train_metrics(run_dir) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [row for row in map(json.loads, lines) if row["section"] == "train"]


def test_the_trainer_moves_logs_and_resumes_the_coefficient(tmp_path):
    from kaisparov.training.trainer import Trainer

    trainer = Trainer(_config(tmp_path, target_entropy=0.3))
    run_id = trainer.train()
    rows = _train_metrics(tmp_path / run_id)
    # Epoch 1 used the configured start; an untrained (near-uniform) policy is above
    # the target, so the thermostat lowered it for epoch 2.
    assert rows[0]["entropy_coef"] == 0.01
    assert rows[1]["entropy_coef"] < 0.01
    reached = trainer.entropy_coef

    checkpoint = tmp_path / run_id / "checkpoints" / "epoch2.pth"
    config = _config(tmp_path, target_entropy=0.3).to_dict()
    resumed = Trainer(TrainConfig.from_dict({**config, "resume_from": str(checkpoint)}))
    assert resumed.entropy_coef == pytest.approx(reached)


def test_a_fixed_coefficient_does_not_move(tmp_path):
    from kaisparov.training.trainer import Trainer

    trainer = Trainer(_config(tmp_path))
    rows = _train_metrics(tmp_path / trainer.train())
    assert [row["entropy_coef"] for row in rows] == [0.01, 0.01]


def test_the_target_must_be_a_fraction(tmp_path):
    from kaisparov.training.trainer import Trainer

    with pytest.raises(ValueError, match="target_entropy"):
        Trainer(_config(tmp_path, target_entropy=1.5))
