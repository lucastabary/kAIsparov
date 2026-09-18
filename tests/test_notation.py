"""Move notation for the in-game history.

Torch-free and pygame-free: everything here runs on the pure-Python engine.
"""

from __future__ import annotations

from kaisparov.core.game import ChessGame
from kaisparov.core.notation import MoveRow, move_to_san, numbered_moves
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player

W, B = Player.WHITE, Player.BLACK


def empty_grid():
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def place(grid, coord, player, piece_type):
    grid[coord[0]][coord[1]] = Piece(player, piece_type)


def game_with(*pieces, turn=W):
    """A game on an otherwise empty board; kings are only added when asked for."""
    grid = empty_grid()
    for coord, player, piece_type in pieces:
        place(grid, coord, player, piece_type)
    return ChessGame(initial_board=grid, turn=turn)


def test_opening_moves_read_like_a_scoresheet():
    game = ChessGame()
    assert move_to_san(game, (4, 1), (4, 3)) == "e4"
    assert move_to_san(game, (6, 0), (5, 2)) == "Cf3"


def test_writing_a_move_down_leaves_the_game_untouched():
    game = ChessGame()
    before = (game.zobrist, game.turn, game.count, list(game.position_history), game.last_move)
    move_to_san(game, (4, 1), (4, 3))
    after = (game.zobrist, game.turn, game.count, list(game.position_history), game.last_move)
    assert after == before
    assert game.grid[4][1] is not None and game.grid[4][3] is None


def test_captures_are_marked_and_pawns_name_their_file():
    game = game_with(
        ((4, 3), W, PieceType.PAWN),
        ((3, 4), B, PieceType.PAWN),
        ((0, 0), W, PieceType.ROOK),
        ((0, 6), B, PieceType.KNIGHT),
    )
    assert move_to_san(game, (4, 3), (3, 4)) == "exd5"
    assert move_to_san(game, (0, 0), (0, 6)) == "Txa7"


def test_en_passant_is_a_capture_even_though_the_square_is_empty():
    game = game_with(((4, 4), W, PieceType.PAWN), ((3, 6), B, PieceType.PAWN), turn=B)
    game.play((3, 6), (3, 4))  # double push, arms en passant on d6
    assert move_to_san(game, (4, 4), (3, 5)) == "exd6"


def test_castling_on_either_side():
    game = game_with(
        ((4, 0), W, PieceType.KING),
        ((7, 0), W, PieceType.ROOK),
        ((0, 0), W, PieceType.ROOK),
    )
    assert move_to_san(game, (4, 0), (6, 0)) == "O-O"
    assert move_to_san(game, (4, 0), (2, 0)) == "O-O-O"


def test_twins_are_told_apart_by_file_then_rank():
    by_file = game_with(((0, 0), W, PieceType.ROOK), ((7, 0), W, PieceType.ROOK))
    assert move_to_san(by_file, (0, 0), (3, 0)) == "Tad1"

    by_rank = game_with(((0, 0), W, PieceType.ROOK), ((0, 6), W, PieceType.ROOK))
    assert move_to_san(by_rank, (0, 0), (0, 3)) == "T1a4"

    # Knights on c3, e3 and c7 all reach d5: e3 shares c3's rank and c7 its file.
    both = game_with(
        ((2, 2), W, PieceType.KNIGHT),
        ((4, 2), W, PieceType.KNIGHT),
        ((2, 6), W, PieceType.KNIGHT),
    )
    assert move_to_san(both, (2, 2), (3, 4)) == "Cc3d5"


def test_a_twin_that_cannot_reach_the_square_needs_no_disambiguation():
    game = game_with(((0, 0), W, PieceType.ROOK), ((7, 7), W, PieceType.ROOK))
    assert move_to_san(game, (0, 0), (3, 0)) == "Td1"


def test_check_and_king_capture_suffixes():
    game = game_with(((0, 0), W, PieceType.QUEEN), ((7, 7), B, PieceType.KING))
    assert move_to_san(game, (0, 0), (0, 7)) == "Da8+"
    assert move_to_san(game, (0, 0), (7, 7)) == "Dxh8#"


def test_plies_pair_into_numbered_rows():
    assert numbered_moves([]) == []
    assert numbered_moves(["e4", "e5", "Cf3"]) == [
        MoveRow(1, "e4", "e5"),
        MoveRow(2, "Cf3", ""),
    ]


def test_a_game_black_opens_starts_with_an_ellipsis():
    assert numbered_moves(["e5", "Cf3", "Cc6"], first=B) == [
        MoveRow(1, "...", "e5"),
        MoveRow(2, "Cf3", "Cc6"),
    ]
