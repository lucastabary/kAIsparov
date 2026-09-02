"""End-to-end smoke test: collect self-play data and run one PPO update.

Deliberately avoids importing the pygame UI so it runs in a headless CI.
"""

from __future__ import annotations

import torch

from kaisparov.core.board import ChessGame
from kaisparov.models.factory import load_backend, load_backend_spec
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum


def test_training_step_runs():
    device = torch.device("cpu")
    module = load_backend("rgcn")
    spec = load_backend_spec("rgcn")

    agent = spec.model_class.create_agent(device=device, hidden_dim=8)
    optimizer = spec.model_class.create_optimizer(agent, learning_rate=1e-3)
    buffer = spec.buffer_class()
    curriculum = PieceCountCurriculum(
        PhaseConfig(name="test", max_pieces_per_side=4, allow_major=False),
        seed=0,
    )

    rollout_stats = spec.collect_data(
        agent,
        ChessGame(),
        buffer,
        num_episodes=2,
        max_steps_per_episode=20,
        model_module=module,
        curriculum=curriculum,
    )
    assert len(buffer) > 0, "self-play collected no transitions"
    for key in ("king_capture_rate", "truncated_rate", "stalemate_rate", "avg_plies"):
        assert key in rollout_stats

    metrics = spec.train_one_epoch(agent, buffer, optimizer, device=device)
    for key in ("policy_loss", "value_loss", "entropy", "loss", "steps"):
        assert key in metrics
    assert metrics["steps"] == len(buffer)


def test_curriculum_places_both_kings():
    from kaisparov.core.pieces import PieceType, Player

    board = PieceCountCurriculum(
        PhaseConfig(name="test", max_pieces_per_side=6), seed=1
    ).get_initial_board()
    pieces = [p for col in board for p in col if p is not None]
    kings = [p for p in pieces if p.type == PieceType.KING]
    assert {p.player for p in kings} == {Player.WHITE, Player.BLACK}
