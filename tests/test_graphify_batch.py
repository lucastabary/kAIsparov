"""graphify_batch must produce exactly the same graphs as per-game graphify."""

from __future__ import annotations

import random

import torch

from kaisparov.core.game import ChessGame
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


def test_graphify_batch_matches_per_game():
    proc = RGCNProcessor()
    games = _collect_games(120)
    batched = proc.graphify_batch(games)
    assert len(batched) == len(games)
    for single_game, batched_data in zip(games, batched, strict=True):
        expected = proc.graphify(single_game)
        assert torch.equal(batched_data.x, expected.x)
        assert torch.equal(batched_data.edge_index, expected.edge_index)
        assert torch.equal(batched_data.edge_type, expected.edge_type)


def test_graphify_batch_empty():
    assert RGCNProcessor().graphify_batch([]) == []
