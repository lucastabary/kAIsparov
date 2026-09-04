"""Tests for the opponent pool and the single-agent (vs opponent) rollout."""

from __future__ import annotations

import pytest
import torch

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.minimax_agent import MinimaxAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.models.factory import load_backend, load_backend_spec
from kaisparov.training.config import RewardSettings
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum
from kaisparov.training.opponents import OpponentPool
from kaisparov.training.rollout_vs import _gain, collect_vs_opponent


def _spec_and_agent():
    spec = load_backend_spec("rgcn")
    agent = spec.model_class.create_agent(device=torch.device("cpu"), hidden_dim=8)
    return spec, agent


def test_gain_material_and_king():
    r = RewardSettings(material=1.0, king_capture=4.0)
    assert _gain(r, None) == 0.0
    assert _gain(r, Piece(Player.BLACK, PieceType.QUEEN)) == 9.0  # queen material
    assert _gain(r, Piece(Player.BLACK, PieceType.KING)) == 4.0  # flat king bonus (decoupled)


def test_pool_snapshot_sample_and_cap():
    spec, agent = _spec_and_agent()
    pool = OpponentPool(spec, torch.device("cpu"), hidden_dim=8, max_size=2, seed=0)
    assert len(pool) == 0 and pool.sample() is None
    for _ in range(3):
        pool.snapshot(agent)
    assert len(pool) == 2  # capped at max_size
    opp = pool.sample()
    assert opp is not None and hasattr(opp, "select_move")
    # A frozen opponent is independent of the live agent (no grad).
    assert all(not p.requires_grad for p in opp.model.parameters())


def test_pool_baselines_available_from_start():
    spec, _ = _spec_and_agent()
    pool = OpponentPool(
        spec, torch.device("cpu"), hidden_dim=8, seed=0, baselines=[RandomAgent(seed=0)]
    )
    # A seeded baseline makes the pool usable before any snapshot exists.
    assert len(pool) == 1
    assert pool.sample() is not None


def test_pool_group_weight_zero_excludes_snapshots():
    # snapshot_weight=0 -> once both groups exist, only baselines are ever drawn,
    # no matter how many snapshots have accumulated (the dilution fix).
    spec, agent = _spec_and_agent()
    material = MaterialAgent(seed=0)
    pool = OpponentPool(
        spec,
        torch.device("cpu"),
        hidden_dim=8,
        seed=0,
        baselines=[material],
        baseline_weight=1.0,
        snapshot_weight=0.0,
    )
    for _ in range(3):
        pool.snapshot(agent)
    assert len(pool._agents) == 3
    assert all(pool.sample() is material for _ in range(30))


def test_pool_baseline_weights_bias():
    # A zero weight on the second baseline means it is never chosen.
    spec, _ = _spec_and_agent()
    material, random_agent = MaterialAgent(seed=0), RandomAgent(seed=0)
    pool = OpponentPool(
        spec,
        torch.device("cpu"),
        hidden_dim=8,
        seed=0,
        baselines=[material, random_agent],
        baseline_weights=[1.0, 0.0],
    )
    assert all(pool.sample() is material for _ in range(30))


def test_pool_baseline_weights_length_mismatch_raises():
    spec, _ = _spec_and_agent()
    with pytest.raises(ValueError):
        OpponentPool(
            spec,
            torch.device("cpu"),
            hidden_dim=8,
            baselines=[MaterialAgent(seed=0), RandomAgent(seed=0)],
            baseline_weights=[1.0],  # only one weight for two baselines
        )


def test_pool_minimax_snapshots():
    # search_depth >= 1 wraps frozen snapshots in a Minimax search.
    spec, agent = _spec_and_agent()
    pool = OpponentPool(spec, torch.device("cpu"), hidden_dim=8, seed=0, search_depth=1)
    pool.snapshot(agent)
    opp = pool.sample()
    assert isinstance(opp, MinimaxAgent) and opp.depth == 1
    assert all(not p.requires_grad for p in opp.model.parameters())


def test_collect_vs_opponent_fills_buffer_and_reports():
    spec, agent = _spec_and_agent()
    module = load_backend("rgcn")
    buffer = spec.buffer_class(self_play=False)
    curriculum = PieceCountCurriculum(
        PhaseConfig(name="t", max_pieces_per_side=4, allow_major=False), seed=0
    )

    stats = collect_vs_opponent(
        agent,
        buffer,
        num_episodes=6,
        max_steps_per_episode=20,
        model_module=module,
        curriculum=curriculum,
        reward_settings=RewardSettings(material=1.0, king_capture=4.0),
        opponent=RandomAgent(seed=1),
        seed=0,
    )
    assert len(buffer) > 0
    assert stats["winrate"] + stats["lossrate"] + stats["drawrate"] == 1.0
    # king_capture_rate = the decisive games (win + loss), mirroring the self-play key.
    assert stats["king_capture_rate"] == pytest.approx(stats["winrate"] + stats["lossrate"])
    buffer.compute_returns_and_advantages()  # standard (single-agent) GAE runs
    assert len(buffer.advantages) == len(buffer)


def test_collect_vs_opponent_samples_per_episode():
    # `sample_opponent` draws a fresh opponent each episode; record how many draws.
    spec, agent = _spec_and_agent()
    module = load_backend("rgcn")
    buffer = spec.buffer_class(self_play=False)

    draws: list[int] = []

    def sampler():
        draws.append(1)
        return RandomAgent(seed=len(draws))

    stats = collect_vs_opponent(
        agent,
        buffer,
        num_episodes=4,
        max_steps_per_episode=10,
        model_module=module,
        curriculum=None,
        reward_settings=RewardSettings(),
        sample_opponent=sampler,
        seed=0,
    )
    assert len(draws) == 4  # one opponent drawn per episode, not once per epoch
    assert "king_capture_rate" in stats


def test_collect_vs_opponent_requires_exactly_one_opponent():
    spec, agent = _spec_and_agent()
    module = load_backend("rgcn")
    buffer = spec.buffer_class(self_play=False)
    kwargs = dict(
        num_episodes=1,
        max_steps_per_episode=5,
        model_module=module,
        curriculum=None,
        reward_settings=RewardSettings(),
    )
    with pytest.raises(ValueError):  # neither given
        collect_vs_opponent(agent, buffer, **kwargs)
    with pytest.raises(ValueError):  # both given
        collect_vs_opponent(
            agent, buffer, opponent=RandomAgent(seed=0), sample_opponent=lambda: None, **kwargs
        )


def test_collect_vs_opponent_standard_start():
    # Also works without a curriculum (games from the normal position).
    spec, agent = _spec_and_agent()
    module = load_backend("rgcn")
    buffer = spec.buffer_class(self_play=False)
    stats = collect_vs_opponent(
        agent,
        buffer,
        num_episodes=2,
        max_steps_per_episode=10,
        model_module=module,
        curriculum=None,
        reward_settings=RewardSettings(),
        opponent=RandomAgent(seed=2),
        seed=0,
    )
    assert "avg_plies" in stats
    assert isinstance(ChessGame(), ChessGame)  # sanity
