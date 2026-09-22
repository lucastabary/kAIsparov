"""graphify_batch must produce exactly the same graphs as per-game graphify."""

from __future__ import annotations

import random

import pytest
import torch

from kaisparov.core.game import ChessGame
from kaisparov.models.features import FEATURE_SETS
from kaisparov.models.rgcn.processor import RGCNProcessor


def _collect_games(n_target: int, seed: int = 3) -> list[ChessGame]:
    rng = random.Random(seed)
    games: list[ChessGame] = []
    while len(games) < n_target:
        game = ChessGame()
        for _ in range(60):
            moves = game.legal_moves()
            games.append(game.copy())
            if len(games) >= n_target or not moves:
                break
            game.make(*rng.choice(moves))
    return games[:n_target]


@pytest.mark.parametrize("features", sorted(FEATURE_SETS))
def test_graphify_batch_matches_per_game(features):
    proc = RGCNProcessor(features=features)
    games = _collect_games(120)
    batched = proc.graphify_batch(games)
    assert len(batched) == len(games)
    for single_game, batched_data in zip(games, batched, strict=True):
        expected = proc.graphify(single_game)
        assert expected.x.shape == (64, FEATURE_SETS[features].dim)
        assert torch.equal(batched_data.x, expected.x)
        assert torch.equal(batched_data.edge_index, expected.edge_index)
        assert torch.equal(batched_data.edge_type, expected.edge_type)


def test_graphify_batch_empty():
    assert RGCNProcessor().graphify_batch([]) == []


@pytest.mark.parametrize("backend", ["rgcn", "shared_rgcn"])
@pytest.mark.parametrize("features", sorted(FEATURE_SETS))
def test_a_model_reads_the_feature_set_it_was_built_for(backend, features):
    from kaisparov.models.factory import load_backend_spec

    spec = load_backend_spec(backend)
    model = spec.model_class(hidden_dim=4, features=features)
    assert model.features == features
    data = spec.processor_class(features=model.features).graphify(ChessGame())
    scores, _value = model(data)  # the input layer is exactly as wide as the features
    assert scores.shape == (data.edge_index.shape[1],)


def test_an_unknown_feature_set_is_refused():
    with pytest.raises(ValueError, match="Unknown node feature set"):
        RGCNProcessor(features="does_not_exist")
