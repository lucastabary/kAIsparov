"""Tests for the MinimaxAgent (search on the critic, ordered by the actor)."""

from __future__ import annotations

import torch

from kaisparov.agents.minimax_agent import MinimaxAgent
from kaisparov.core.board import ChessGame
from kaisparov.core.coords import BOARD_SIZE, all_squares
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.models.factory import load_backend_spec


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def place(game, coord, player, piece_type):
    game.grid[coord[0]][coord[1]] = Piece(player, piece_type)


def serialize(game):
    return tuple(
        None if (p := game.grid[x][y]) is None else (p.player, p.type, p.has_moved)
        for x, y in all_squares()
    )


def make_agent(depth: int) -> MinimaxAgent:
    spec = load_backend_spec("rgcn")
    model = spec.model_class.create_agent(device=torch.device("cpu"), hidden_dim=8)
    return MinimaxAgent(model, spec.processor_class(), depth=depth)


def test_minimax_captures_the_king_when_available():
    # King capture = WIN dominates any critic value, so the agent must take it,
    # regardless of the (untrained) network.
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (3, 0), Player.BLACK, PieceType.KING)
    place(game, (7, 7), Player.WHITE, PieceType.KING)

    move = make_agent(depth=2).select_move(game)
    assert move == ((0, 0), (3, 0))  # rook takes the king


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
    source, dest = move
    assert dest in game.possible_moves(source)
