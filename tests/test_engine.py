"""Correctness tests for the chess engine: movegen, make/unmake, and perft."""

from __future__ import annotations

from kaisparov.core.board import ChessGame
from kaisparov.core.coords import BOARD_SIZE, all_squares
from kaisparov.core.pieces import Piece, PieceType, Player


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def place(game: ChessGame, coord, player: Player, piece_type: PieceType) -> None:
    game.grid[coord[0]][coord[1]] = Piece(player, piece_type)


def serialize(game: ChessGame):
    return tuple(
        None if (p := game.grid[x][y]) is None else (p.player, p.type, p.has_moved)
        for x, y in all_squares()
    )


# --------------------------------------------------------------- move generation
def test_knight_moves_center_and_corner():
    game = empty_game()
    place(game, (3, 3), Player.WHITE, PieceType.KNIGHT)
    assert len(game.possible_moves((3, 3))) == 8

    game2 = empty_game()
    place(game2, (0, 0), Player.WHITE, PieceType.KNIGHT)
    assert set(game2.possible_moves((0, 0))) == {(1, 2), (2, 1)}


def test_sliding_counts_on_empty_board():
    game = empty_game()
    place(game, (3, 3), Player.WHITE, PieceType.ROOK)
    assert len(game.possible_moves((3, 3))) == 14
    game.grid[3][3] = None

    place(game, (0, 0), Player.WHITE, PieceType.BISHOP)
    assert len(game.possible_moves((0, 0))) == 7
    game.grid[0][0] = None

    place(game, (3, 3), Player.WHITE, PieceType.QUEEN)
    assert len(game.possible_moves((3, 3))) == 27


def test_sliding_blocked_by_own_and_captures_enemy():
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (0, 3), Player.WHITE, PieceType.PAWN)  # own piece blocks the file
    place(game, (3, 0), Player.BLACK, PieceType.PAWN)  # enemy is capturable
    dests = set(game.possible_moves((0, 0)))
    assert (0, 3) not in dests and (0, 4) not in dests  # blocked before/at own piece
    assert (3, 0) in dests and (4, 0) not in dests  # capture, then stop


def test_pawn_pushes_and_captures():
    game = empty_game()
    place(game, (4, 1), Player.WHITE, PieceType.PAWN)
    assert set(game.possible_moves((4, 1))) == {(4, 2), (4, 3)}  # single + double

    place(game, (5, 2), Player.BLACK, PieceType.PAWN)
    assert (5, 2) in set(game.possible_moves((4, 1)))  # diagonal capture


def test_castling_offered_when_path_clear():
    game = empty_game()
    place(game, (4, 0), Player.WHITE, PieceType.KING)
    place(game, (7, 0), Player.WHITE, PieceType.ROOK)
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    dests = set(game.possible_moves((4, 0)))
    assert (6, 0) in dests  # kingside
    assert (2, 0) in dests  # queenside


def test_no_castling_from_nonstandard_king_square():
    # King off its home file: castling must not be offered even with unmoved rooks
    # (guards the random curriculum positions).
    game = empty_game()
    place(game, (2, 0), Player.WHITE, PieceType.KING)
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (7, 0), Player.WHITE, PieceType.ROOK)
    dests = set(game.possible_moves((2, 0)))
    assert (0, 0) not in dests and (4, 0) not in dests  # no two-square king jump


# ---------------------------------------------------------------------- en passant
def test_en_passant_capture():
    game = empty_game()  # White to move
    place(game, (4, 1), Player.WHITE, PieceType.PAWN)
    place(game, (3, 3), Player.BLACK, PieceType.PAWN)

    game.play((4, 1), (4, 3))  # White double push
    assert game.en_passant_target == (4, 2)
    assert (4, 2) in game.possible_moves((3, 3))  # Black may capture en passant

    captured = game.play((3, 3), (4, 2))  # Black takes en passant
    assert captured is not None and captured.type == PieceType.PAWN
    assert captured.player == Player.WHITE
    assert game.grid[4][3] is None  # the double-pushed pawn is removed
    assert game.grid[4][2] is not None and game.grid[4][2].player == Player.BLACK
    assert game.grid[3][3] is None


def test_en_passant_make_unmake_restores_state():
    game = empty_game()
    place(game, (4, 1), Player.WHITE, PieceType.PAWN)
    place(game, (3, 3), Player.BLACK, PieceType.PAWN)
    game.play((4, 1), (4, 3))  # arm en passant

    before = serialize(game)
    ep_before = game.en_passant_target
    undo = game.make((3, 3), (4, 2))  # en passant capture
    assert game.grid[4][3] is None
    game.unmake(undo)
    assert serialize(game) == before
    assert game.en_passant_target == ep_before
    assert game.turn == Player.BLACK


def test_en_passant_only_available_immediately():
    game = empty_game()
    place(game, (4, 1), Player.WHITE, PieceType.PAWN)
    place(game, (3, 3), Player.BLACK, PieceType.PAWN)
    place(game, (0, 6), Player.BLACK, PieceType.KING)
    place(game, (0, 0), Player.WHITE, PieceType.KING)

    game.play((4, 1), (4, 3))  # White double push -> en passant armed
    game.play((0, 6), (0, 5))  # Black plays elsewhere -> window closes
    assert game.en_passant_target is None
    game.play((0, 0), (0, 1))  # White plays; back to Black
    assert (4, 2) not in game.possible_moves((3, 3))  # en passant no longer legal


# ------------------------------------------------------------------ make / unmake
def test_make_unmake_restores_state():
    game = ChessGame()
    before = serialize(game)
    undo = game.make((4, 1), (4, 3))  # e2-e4-style double push
    assert serialize(game) != before
    assert game.turn == Player.BLACK
    game.unmake(undo)
    assert serialize(game) == before
    assert game.turn == Player.WHITE
    assert game.count == 0


def test_castling_make_unmake_moves_and_restores_rook():
    game = empty_game()
    place(game, (4, 0), Player.WHITE, PieceType.KING)
    place(game, (7, 0), Player.WHITE, PieceType.ROOK)
    before = serialize(game)
    undo = game.make((4, 0), (6, 0))  # kingside castle
    assert game.grid[5][0] is not None and game.grid[5][0].type == PieceType.ROOK
    assert game.grid[7][0] is None
    game.unmake(undo)
    assert serialize(game) == before


def test_copy_is_independent():
    game = ChessGame()
    clone = game.copy()
    clone.make((4, 1), (4, 3))
    assert serialize(game) != serialize(clone)
    assert game.turn == Player.WHITE  # original untouched


# -------------------------------------------------------------------------- perft
def _perft(game: ChessGame, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    player = game.turn
    for src in all_squares():
        piece = game.grid[src[0]][src[1]]
        if piece is None or piece.player != player:
            continue
        for dest in game.possible_moves(src):
            undo = game.make(src, dest)
            total += _perft(game, depth - 1)
            game.unmake(undo)
    return total


def test_perft_matches_standard_chess_from_start():
    # No captures/checks/castling occur within these depths from the initial
    # position, so the counts coincide with standard chess perft.
    game = ChessGame()
    assert _perft(game, 1) == 20
    assert _perft(game, 2) == 400
    assert _perft(game, 3) == 8902
