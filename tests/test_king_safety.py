"""Tests for the optional blunder filter (kaisparov.agents.safety) and the
``avoid_king_suicide`` flag on the baseline agents.

Standard chess makes hanging your own king illegal, so the guard now protects
against the next blunder up: playing a move after which the opponent mates at once.
The recurring scenario is a back-rank mate the greedy agent walks into while
grabbing a free piece.
"""

from __future__ import annotations

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.agents.safety import safe_moves, walks_into_mate
from kaisparov.core.game import ChessGame
from kaisparov.core.move import Move


def game_from(fen: str) -> ChessGame:
    import chess

    return ChessGame(board=chess.Board(fen))


# White to move. Taking the black knight on a5 with the rook is the only capture and
# the greedy choice, but it abandons the first rank and Black mates with ...Re1#.
# Everything else (h2h3 among them) keeps the escape square and survives.
BAIT_FEN = "4r1k1/5ppp/8/n7/8/8/R4PPP/6K1 w - - 0 1"
GREEDY_MOVE = Move((0, 1), (0, 4))  # Ra2xa5
SAFE_MOVE = Move((7, 1), (7, 2))  # h2h3, making luft


def test_the_bait_capture_is_legal_and_greedy():
    game = game_from(BAIT_FEN)
    moves = game.legal_moves()
    assert GREEDY_MOVE in moves
    assert MaterialAgent(seed=0).select_move(game) == GREEDY_MOVE


def test_walks_into_mate_detects_the_back_rank():
    game = game_from(BAIT_FEN)
    assert walks_into_mate(game, GREEDY_MOVE) is True


def test_walks_into_mate_false_for_a_move_that_makes_luft():
    game = game_from(BAIT_FEN)
    assert walks_into_mate(game, SAFE_MOVE) is False


def test_safe_moves_drops_the_blunder_but_keeps_the_rest():
    game = game_from(BAIT_FEN)
    moves = game.legal_moves()
    filtered = safe_moves(game, moves)
    assert GREEDY_MOVE in moves  # it is a legal candidate...
    assert GREEDY_MOVE not in filtered  # ...but the guard removes it
    assert SAFE_MOVE in filtered


def test_safe_moves_falls_back_when_every_move_loses():
    # Black to move with a lone king on g8 and two squares to run to; the queen on c7
    # covers the 7th rank and the rook on a3 swings to a8, so both f8 and h8 are mated
    # next ply. Nothing is safe, so the guard must not hand back an empty list.
    game = game_from("6k1/2Q5/8/8/2K5/R7/8/8 b - - 0 1")
    moves = game.legal_moves()
    assert moves  # Black still has moves...
    assert all(walks_into_mate(game, m) for m in moves)  # ...all of them lose
    assert safe_moves(game, moves) == moves  # so we fall back to the full list


def test_material_agent_refuses_the_bait_with_the_guard():
    game = game_from(BAIT_FEN)
    move = MaterialAgent(seed=0, avoid_king_suicide=True).select_move(game)
    assert move != GREEDY_MOVE
    assert not walks_into_mate(game, move)


def test_random_agent_never_walks_into_mate_with_the_guard():
    agent = RandomAgent(seed=0, avoid_king_suicide=True)
    # Draw many moves; the guarded agent must never allow the back-rank mate.
    assert all(agent.select_move(game_from(BAIT_FEN)) != GREEDY_MOVE for _ in range(50))
