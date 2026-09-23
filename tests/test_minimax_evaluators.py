"""Tests for MinimaxAgent on the handcrafted evaluators (material, heuristic),
and for the Fallible wrapper that makes any agent blunder now and then."""

from __future__ import annotations

import pytest

from kaisparov.agents.fallible import Fallible
from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.minimax_agent import MinimaxAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.analysis.evaluators import HeuristicEvaluator, MaterialEvaluator
from kaisparov.bench.position import Position
from kaisparov.core.game import ChessGame
from kaisparov.core.move import Move


def _game(fen: str) -> ChessGame:
    return Position(fen).to_game()


def _uci(move) -> str:
    return Move.coerce(move).uci()


def _material(depth: int = 2, seed: int = 0) -> MinimaxAgent:
    return MinimaxAgent(MaterialEvaluator(), depth=depth, seed=seed)


def test_takes_a_hung_queen():
    game = _game("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1")
    assert _uci(_material().select_move(game)) == "d1d5"


def test_refuses_a_poisoned_capture_the_greedy_baseline_takes():
    # Qxd5 wins a pawn and loses the queen to cxd5.
    fen = "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1"
    assert _uci(MaterialAgent(seed=0).select_move(_game(fen))) == "d1d5"
    for seed in range(5):
        assert _uci(_material(seed=seed).select_move(_game(fen))) != "d1d5"


def test_finds_mate_in_one_even_at_depth_one():
    game = _game("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert _uci(_material(depth=1).select_move(game)) == "a1a8"


def test_a_queen_up_it_never_stalemates():
    # Qg6 and Qf7 stalemate the bare king: worth 0, not the queen it is up.
    for seed in range(10):
        game = _game("7k/8/8/5Q2/8/8/8/6K1 w - - 0 1")
        game.make(*_material(seed=seed).select_move(game))
        assert not game.is_stalemate()


@pytest.mark.parametrize("evaluator", [MaterialEvaluator(), HeuristicEvaluator()])
def test_leaves_the_position_as_it_found_it(evaluator):
    game = _game("r3k2r/ppp2ppp/2n5/3qp3/3P4/2N5/PPP2PPP/R2QK2R w KQkq - 0 1")
    before = game.fen()
    MinimaxAgent(evaluator, depth=2, seed=0).select_move(game)
    assert game.fen() == before


def test_the_incremental_material_score_matches_a_recount():
    # The fast path (MaterialEvaluator.move_delta) must choose what a full recount at
    # every leaf chooses: same search, the evaluator stripped of its hook.
    class Recount:
        name, slope, default_lookahead = "recount", 0.368, 1

        def evaluate(self, game):
            return MaterialEvaluator().evaluate(game)

    fens = [
        "4k3/8/2p5/3p4/8/8/8/3QK3 w - - 0 1",
        "r3k2r/ppp2ppp/2n5/3qp3/3P4/2N5/PPP2PPP/R2QK2R w KQkq - 0 1",
        "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
    ]
    for fen in fens:
        game = _game(fen)
        fast, slow = _material(), MinimaxAgent(Recount(), depth=2, seed=0)
        moves = game.legal_moves()
        assert [
            fast._child(game, m, 1, -1e9, 1e9, MaterialEvaluator().evaluate(game), 1) for m in moves
        ] == [slow._child(game, m, 1, -1e9, 1e9, 0.0, 1) for m in moves]


def test_the_heuristic_search_prefers_the_centre_on_a_quiet_board():
    # Nothing to win at the start: material rates every move alike, the heuristic
    # does not, and its search picks a move that improves a piece.
    game = ChessGame()
    assert _uci(MinimaxAgent(HeuristicEvaluator(), depth=1, seed=0).select_move(game)) in {
        "e2e4",
        "d2d4",
        "b1c3",
        "g1f3",
        "e2e3",
        "d2d3",
        "b1d2",
        "g1e2",
    }


def test_depth_zero_is_refused():
    with pytest.raises(ValueError, match="depth >= 1"):
        MinimaxAgent(MaterialEvaluator(), depth=0)


# ------------------------------------------------------------------------ Fallible


def test_fallible_zero_plays_the_agent_one_plays_random():
    fen = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1"
    never = Fallible(_material(), 0.0, seed=0)
    assert all(_uci(never.select_move(_game(fen))) == "d1d5" for _ in range(20))
    always = Fallible(_material(), 1.0, seed=0)
    moves = {_uci(always.select_move(_game(fen))) for _ in range(60)}
    assert len(moves) > 3  # random legal moves, not the search's pick


def test_fallible_blunders_at_the_rate_it_is_given():
    fen = "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1"
    agent = Fallible(_material(), 0.3, seed=1)
    n = 2000
    # A random move is the queen capture 1 time in 14 (legal moves here).
    n_moves = len(_game(fen).legal_moves())
    other = sum(_uci(agent.select_move(_game(fen))) != "d1d5" for _ in range(n)) / n
    assert other == pytest.approx(0.3 * (n_moves - 1) / n_moves, abs=0.03)


def test_fallible_names_itself_and_checks_its_probability():
    assert Fallible(RandomAgent(), 0.1).name == "random+random0.1"
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Fallible(RandomAgent(), 1.5)
