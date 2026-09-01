"""Tests for baseline policies."""

from __future__ import annotations

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.core.board import ChessGame
from kaisparov.core.coords import BOARD_SIZE
from kaisparov.core.pieces import Piece, PieceType, Player


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def place(game, coord, player, piece_type):
    game.grid[coord[0]][coord[1]] = Piece(player, piece_type)


def test_random_agent_returns_legal_move_and_is_seeded():
    game = ChessGame()
    a = RandomAgent(seed=42)
    move = a.select_move(game)
    assert move is not None
    source, dest = move
    assert dest in game.possible_moves(source)
    # Same seed -> same first move.
    assert RandomAgent(seed=42).select_move(ChessGame()) == move


def test_random_agent_none_when_no_pieces():
    game = empty_game(turn=Player.WHITE)
    place(game, (4, 4), Player.BLACK, PieceType.KING)  # only enemy pieces
    assert RandomAgent(seed=0).select_move(game) is None


def test_material_agent_grabs_hanging_queen():
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (0, 5), Player.BLACK, PieceType.QUEEN)
    assert MaterialAgent(seed=0).select_move(game) == ((0, 0), (0, 5))


def test_material_agent_prefers_capturing_the_king():
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (0, 3), Player.BLACK, PieceType.PAWN)  # low-value capture on the file
    place(game, (3, 0), Player.BLACK, PieceType.KING)  # king capture on the rank
    assert MaterialAgent(seed=0).select_move(game) == ((0, 0), (3, 0))
