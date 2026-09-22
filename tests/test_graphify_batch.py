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


def test_the_policy_can_castle_and_promote():
    """Both are reachable in the action space; castling rides on the rook relation.

    A king castles two squares along its rank, which the rook relation already has as
    an edge, so the mask keeps it. Underpromotion does not: the four promotions share
    one (source, dest) pair, and playing it queens.
    """
    import chess

    from kaisparov.core.utils import coord_to_index

    processor = RGCNProcessor()

    def is_reachable(fen, source, dest):
        game = ChessGame(board=chess.Board(fen))
        edge_index = processor.static_graph_edges[0]
        legal = processor.legal_mask(game).nonzero().flatten().tolist()
        keys = {int(edge_index[0, k]) * 64 + int(edge_index[1, k]) for k in legal}
        return coord_to_index(source) * 64 + coord_to_index(dest) in keys

    castle = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    assert is_reachable(castle, (4, 0), (6, 0))  # O-O
    assert is_reachable(castle, (4, 0), (2, 0))  # O-O-O
    assert is_reachable("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", (0, 6), (0, 7))  # promotion
