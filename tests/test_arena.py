"""Tests for the evaluation arena."""

from __future__ import annotations

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.core.pieces import Player
from kaisparov.eval.arena import evaluate, play_game


def test_play_game_terminates_with_valid_result():
    result = play_game(RandomAgent(seed=1), RandomAgent(seed=2), max_plies=200)
    assert result.winner in (Player.WHITE, Player.BLACK, None)
    assert result.plies > 0


def test_evaluate_tallies_add_up():
    stats = evaluate(RandomAgent(seed=1), RandomAgent(seed=2), games=10, max_plies=120)
    assert stats.wins_a + stats.wins_b + stats.draws == stats.games


def test_material_beats_random():
    stats = evaluate(MaterialAgent(seed=0), RandomAgent(seed=0), games=40, max_plies=200)
    assert stats.wins_a > stats.wins_b
    assert stats.score_a > 0.5
