"""Tests for MaterialMinimaxAgent (alpha-beta search scored on material)."""

from __future__ import annotations

import pytest

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.material_minimax import MaterialMinimaxAgent
from kaisparov.bench.position import Position
from kaisparov.core.move import Move


def _game(fen: str):
    return Position(fen).to_game()


def _uci(move) -> str:
    return Move.coerce(move).uci()


def test_takes_a_hung_queen():
    game = _game("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert _uci(MaterialMinimaxAgent(depth=2, seed=0).select_move(game)) == "d1d5"


def test_refuses_a_poisoned_capture_the_greedy_baseline_takes():
    # Qxd5 wins a pawn and loses the queen to cxd5.
    fen = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
    assert _uci(MaterialAgent(seed=0).select_move(_game(fen))) == "d1d5"
    for seed in range(5):
        assert _uci(MaterialMinimaxAgent(depth=2, seed=seed).select_move(_game(fen))) != "d1d5"


def test_finds_mate_in_one_even_at_depth_one():
    game = _game("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert _uci(MaterialMinimaxAgent(depth=1, seed=0).select_move(game)) == "a1a8"


def test_a_queen_up_it_never_stalemates():
    # Qg6 and Qf7 stalemate the bare king: worth 0, not the queen it is up.
    for seed in range(10):
        game = _game("7k/8/8/5Q2/8/8/8/6K1 w - - 0 1")
        game.make(*MaterialMinimaxAgent(depth=2, seed=seed).select_move(game))
        assert not game.is_stalemate()


def test_leaves_the_position_as_it_found_it():
    game = _game("r3k2r/ppp2ppp/2n5/3qp3/3P4/2N5/PPP2PPP/R2QK2R w KQkq - 0 1")
    before = game.fen()
    MaterialMinimaxAgent(depth=2, seed=0).select_move(game)
    assert game.fen() == before


def test_depth_zero_is_refused():
    with pytest.raises(ValueError, match="depth >= 1"):
        MaterialMinimaxAgent(depth=0)
