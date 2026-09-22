"""Tests for baseline policies."""

from __future__ import annotations

from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent
from kaisparov.core.coords import BOARD_SIZE
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import Piece, PieceType, Player


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def place(game, coord, player, piece_type):
    game.place(coord, Piece(player, piece_type))


def test_random_agent_returns_legal_move_and_is_seeded():
    game = ChessGame()
    a = RandomAgent(seed=42)
    move = a.select_move(game)
    assert move is not None
    source, dest = move[0], move[1]
    assert dest in game.possible_moves(source)
    # Same seed -> same first move.
    assert RandomAgent(seed=42).select_move(ChessGame()) == move


def test_random_agent_none_when_no_pieces():
    game = empty_game(turn=Player.WHITE)
    place(game, (4, 4), Player.BLACK, PieceType.KING)  # only enemy pieces
    assert RandomAgent(seed=0).select_move(game) is None


def test_material_agent_grabs_hanging_queen():
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)  # a1, the a-file is clear
    place(game, (0, 5), Player.BLACK, PieceType.QUEEN)  # a6, hanging
    place(game, (4, 0), Player.WHITE, PieceType.KING)
    place(game, (7, 7), Player.BLACK, PieceType.KING)
    assert MaterialAgent(seed=0).select_move(game)[:2] == ((0, 0), (0, 5))


def test_material_agent_prefers_the_biggest_capture():
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (0, 3), Player.BLACK, PieceType.PAWN)  # low-value capture on the file
    place(game, (3, 0), Player.BLACK, PieceType.QUEEN)  # the big one, on the rank
    place(game, (4, 4), Player.WHITE, PieceType.KING)
    place(game, (7, 7), Player.BLACK, PieceType.KING)
    assert MaterialAgent(seed=0).select_move(game)[:2] == ((0, 0), (3, 0))


def test_material_agent_queens_a_pawn_over_taking_a_knight():
    game = empty_game()
    place(game, (0, 6), Player.WHITE, PieceType.PAWN)  # a7, one push from queening
    place(game, (1, 7), Player.BLACK, PieceType.KNIGHT)  # b8, capturable by the pawn
    place(game, (4, 0), Player.WHITE, PieceType.KING)
    place(game, (6, 4), Player.BLACK, PieceType.KING)
    move = MaterialAgent(seed=0).select_move(game)
    # Taking the knight and promoting is worth 3 + 8; pushing straight is worth 8.
    assert move[2] is PieceType.QUEEN


def test_material_agent_counts_an_en_passant_capture():
    # exd6 e.p. is the only capture, and it lands on an empty square.
    import chess

    from kaisparov.core.game import ChessGame
    from kaisparov.core.move import Move

    game = ChessGame(board=chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"))
    en_passant = Move((4, 4), (3, 5))
    assert all(MaterialAgent(seed=s).select_move(game) == en_passant for s in range(20))
