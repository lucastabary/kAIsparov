"""Tests for MinimaxAgent.on_model (search on the critic, ordered by the actor)."""

from __future__ import annotations

import torch

from kaisparov.agents.minimax_agent import MinimaxAgent
from kaisparov.core.coords import BOARD_SIZE, all_squares
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.models.factory import load_backend_spec


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def place(game, coord, player, piece_type):
    game.place(coord, Piece(player, piece_type))


def serialize(game):
    return tuple(
        None if (p := game.grid[x][y]) is None else (p.player, p.type, p.has_moved)
        for x, y in all_squares()
    )


def make_agent(depth: int) -> MinimaxAgent:
    spec = load_backend_spec("rgcn")
    model = spec.model_class.create_agent(device=torch.device("cpu"), hidden_dim=8)
    return MinimaxAgent.on_model(model, spec.processor_class(), depth=depth)


def test_minimax_plays_mate_when_available():
    # Mate = WIN dominates any critic value, so the agent must play it, regardless of
    # the (untrained) network. Back-rank mate: Ra1-a8#, the black king shut in by its
    # own pawns on g7/h7.
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (7, 7), Player.BLACK, PieceType.KING)
    place(game, (6, 6), Player.BLACK, PieceType.PAWN)
    place(game, (7, 6), Player.BLACK, PieceType.PAWN)
    place(game, (4, 0), Player.WHITE, PieceType.KING)

    move = make_agent(depth=2).select_move(game)
    assert move[:2] == ((0, 0), (0, 7))  # Ra8#


def test_minimax_has_no_side_effects():
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (4, 4), Player.WHITE, PieceType.KNIGHT)
    place(game, (7, 7), Player.BLACK, PieceType.KING)
    place(game, (0, 7), Player.WHITE, PieceType.KING)

    before, turn = serialize(game), game.turn
    make_agent(depth=2).select_move(game)
    assert serialize(game) == before  # make/unmake restored everything
    assert game.turn == turn


def test_minimax_returns_a_legal_move():
    game = ChessGame()  # standard start
    move = make_agent(depth=1).select_move(game)
    assert move is not None
    source, dest = move[0], move[1]
    assert dest in game.possible_moves(source)


def test_minimax_scores_a_stalemate_as_a_draw():
    # Black to move, not in check, and with no legal move: stalemate. Without the
    # explicit rule the search would read "no move at all" and score the position
    # statically, or worse hand White a win for it.
    game = empty_game(turn=Player.BLACK)
    place(game, (0, 7), Player.BLACK, PieceType.KING)  # a8
    place(game, (2, 6), Player.WHITE, PieceType.QUEEN)  # c7 covers a7/b7/b8
    place(game, (7, 0), Player.WHITE, PieceType.KING)

    assert not game.is_in_check(Player.BLACK)
    assert game.legal_moves() == []
    assert make_agent(depth=2)._search(game, 2, -1e9, 1e9) == 0.0
