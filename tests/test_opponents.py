"""Tests for the opponent pool and the single-agent (vs opponent) rollout."""

from __future__ import annotations

import pytest
import torch

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.minimax_agent import MinimaxAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.core.game import ChessGame
from kaisparov.models.architecture import Architecture
from kaisparov.models.factory import load_backend, load_backend_spec
from kaisparov.training.config import RewardSettings
from kaisparov.training.curriculum import PhaseConfig, PieceCountCurriculum
from kaisparov.training.opponents import OpponentPool
from kaisparov.training.reward import weighted_gain
from kaisparov.training.rollout_vs import collect_vs_opponent

ARCH = Architecture(model="rgcn", hidden_dim=8)


def _spec_and_agent():
    spec = load_backend_spec("rgcn")
    agent = spec.model_class.create_agent(device=torch.device("cpu"), hidden_dim=8)
    return spec, agent


def test_gain_counts_captures_and_promotions():
    import chess

    from kaisparov.core.game import ChessGame

    r = RewardSettings(material=1.0, promotion=1.0, checkmate=4.0)
    assert weighted_gain(r, None) == 0.0

    # Rook takes the queen on a6.
    game = ChessGame(board=chess.Board("7k/8/q7/8/8/8/8/R3K3 w - - 0 1"))
    assert weighted_gain(r, game.make((0, 0), (0, 5))) == 9.0

    # Queening: the pawn leaves, a queen arrives.
    game = ChessGame(board=chess.Board("8/P7/8/8/8/8/8/4K1k1 w - - 0 1"))
    assert weighted_gain(r, game.make((0, 6), (0, 7))) == 8.0

    # The win bonus is not part of weighted_gain: mate is a property of the position, and
    # the caller adds it.
    game = ChessGame(board=chess.Board("7k/6pp/8/8/8/8/8/R3K3 w - - 0 1"))
    undo = game.make((0, 0), (0, 7))
    assert game.is_checkmate()
    assert weighted_gain(r, undo) == 0.0


def test_pool_snapshot_sample_and_cap():
    spec, agent = _spec_and_agent()
    pool = OpponentPool(ARCH, torch.device("cpu"), max_size=2, seed=0)
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
    pool = OpponentPool(ARCH, torch.device("cpu"), seed=0, baselines=[RandomAgent(seed=0)])
    # A seeded baseline makes the pool usable before any snapshot exists.
    assert len(pool) == 1
    assert pool.sample() is not None


def test_pool_group_weight_zero_excludes_snapshots():
    # snapshot_weight=0 -> once both groups exist, only baselines are ever drawn,
    # no matter how many snapshots have accumulated (the dilution fix).
    spec, agent = _spec_and_agent()
    material = MaterialAgent(seed=0)
    pool = OpponentPool(
        ARCH,
        torch.device("cpu"),
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
        ARCH,
        torch.device("cpu"),
        seed=0,
        baselines=[material, random_agent],
        baseline_weights=[1.0, 0.0],
    )
    assert all(pool.sample() is material for _ in range(30))


def test_pool_baseline_weights_length_mismatch_raises():
    spec, _ = _spec_and_agent()
    with pytest.raises(ValueError):
        OpponentPool(
            ARCH,
            torch.device("cpu"),
            baselines=[MaterialAgent(seed=0), RandomAgent(seed=0)],
            baseline_weights=[1.0],  # only one weight for two baselines
        )


def test_pool_minimax_snapshots():
    # search_depth >= 1 wraps frozen snapshots in a Minimax search.
    spec, agent = _spec_and_agent()
    pool = OpponentPool(ARCH, torch.device("cpu"), seed=0, search_depth=1)
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
        reward_settings=RewardSettings(material=1.0, checkmate=4.0),
        opponent=RandomAgent(seed=1),
        seed=0,
    )
    assert len(buffer) > 0
    assert stats["winrate"] + stats["lossrate"] + stats["drawrate"] == 1.0
    # checkmate_rate = the decisive games (win + loss), mirroring the self-play key.
    assert stats["checkmate_rate"] == pytest.approx(stats["winrate"] + stats["lossrate"])
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
    assert "checkmate_rate" in stats


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


def test_pool_mode_scores_the_learner_like_self_play(monkeypatch):
    # Every shaping term (check included) reaches the learner through make_reward_fn;
    # pool mode used to rebuild part of it by hand and silently drop the check bonus.
    import kaisparov.training.rollout_vs as rollout_vs
    from kaisparov.core.pieces import Piece, PieceType, Player

    monkeypatch.setattr(rollout_vs, "make_reward_fn", lambda settings: lambda game, undo: 7.0)

    class BareKings:  # the learner's first move leaves a dead draw: one transition
        def get_initial_board(self):
            grid = [[None] * 8 for _ in range(8)]
            grid[0][0] = Piece(Player.WHITE, PieceType.KING)
            grid[7][7] = Piece(Player.BLACK, PieceType.KING)
            return grid

    spec, agent = _spec_and_agent()
    buffer = spec.buffer_class(gamma=0.99, gae_lambda=0.95, self_play=False)
    collect_vs_opponent(
        agent,
        buffer,
        num_episodes=1,
        max_steps_per_episode=4,
        model_module=load_backend("rgcn"),
        curriculum=BareKings(),
        reward_settings=RewardSettings(),
        opponent=RandomAgent(seed=0),
        seed=0,
    )
    assert buffer.rewards == [7.0]


def test_collect_vs_opponent_seats_the_learner_on_the_strong_side():
    """A lopsided curriculum position is a lesson for its strong side only."""
    spec, agent = _spec_and_agent()
    module = load_backend("rgcn")
    buffer = spec.buffer_class(self_play=False)
    curriculum = PieceCountCurriculum(
        PhaseConfig(
            name="won",
            max_pieces_per_side=3,
            allow_minor=False,
            allow_pawns=False,
            defender_pieces=1,
        ),
        seed=0,
    )

    class Spy(RandomAgent):
        """Checks it is always the bare king that the opponent moves."""

        def select_move(self, game):
            own = [p for column in game.grid for p in column if p and p.player == game.turn]
            assert len(own) == 1  # a lone king
            return super().select_move(game)

    stats = collect_vs_opponent(
        agent,
        buffer,
        num_episodes=8,
        max_steps_per_episode=20,
        model_module=module,
        curriculum=curriculum,
        reward_settings=RewardSettings(checkmate=1.0),
        opponent=Spy(seed=1),
        seed=0,
    )
    assert stats["lossrate"] == 0.0  # a bare king can never mate the learner
