"""Correctness tests for the board facade: movegen, make/unmake, promotion, perft.

The engine itself is python-chess; what these pin down is our wrapper — the
(col, row) convention, the grid snapshot, make/unmake nesting, and the perft counts
that catch a coordinate mix-up immediately.
"""

from __future__ import annotations

from kaisparov.core.coords import BOARD_SIZE, all_squares
from kaisparov.core.game import ChessGame
from kaisparov.core.move import PROMOTION_PIECES
from kaisparov.core.pieces import Piece, PieceType, Player


def empty_game(turn: Player = Player.WHITE) -> ChessGame:
    grid = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    return ChessGame(initial_board=grid, turn=turn)


def place(game: ChessGame, coord, player: Player, piece_type: PieceType) -> None:
    game.place(coord, Piece(player, piece_type))


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
    game.place((3, 3), None)

    place(game, (0, 0), Player.WHITE, PieceType.BISHOP)
    assert len(game.possible_moves((0, 0))) == 7
    game.place((0, 0), None)

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
    place(game, (0, 0), Player.WHITE, PieceType.KING)
    place(game, (0, 6), Player.BLACK, PieceType.KING)
    assert set(game.possible_moves((4, 1))) == {(4, 2), (4, 3)}  # single + double

    place(game, (5, 2), Player.BLACK, PieceType.PAWN)
    assert (5, 2) in set(game.possible_moves((4, 1)))  # diagonal capture


def test_castling_offered_when_path_clear():
    game = empty_game()
    place(game, (4, 0), Player.WHITE, PieceType.KING)
    place(game, (7, 0), Player.WHITE, PieceType.ROOK)
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    place(game, (4, 7), Player.BLACK, PieceType.KING)
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
    place(game, (4, 7), Player.BLACK, PieceType.KING)
    dests = set(game.possible_moves((2, 0)))
    assert (0, 0) not in dests and (4, 0) not in dests  # no two-square king jump


# ---------------------------------------------------------------------- en passant
def test_en_passant_capture():
    game = empty_game()  # White to move
    place(game, (4, 1), Player.WHITE, PieceType.PAWN)
    place(game, (3, 3), Player.BLACK, PieceType.PAWN)
    place(game, (0, 0), Player.WHITE, PieceType.KING)
    place(game, (0, 6), Player.BLACK, PieceType.KING)

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
    place(game, (0, 0), Player.WHITE, PieceType.KING)
    place(game, (0, 6), Player.BLACK, PieceType.KING)
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


def test_last_move_tracked_and_reversible():
    game = ChessGame()
    assert game.last_move is None
    undo = game.make((4, 1), (4, 3))
    assert game.last_move == ((4, 1), (4, 3))
    game.unmake(undo)
    assert game.last_move is None


def test_castling_make_unmake_moves_and_restores_rook():
    game = empty_game()
    place(game, (4, 0), Player.WHITE, PieceType.KING)
    place(game, (7, 0), Player.WHITE, PieceType.ROOK)
    place(game, (0, 7), Player.BLACK, PieceType.KING)
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
    """Node count through our own make/unmake, not python-chess's.

    Going through the facade is the whole point: a mistake in the coordinate
    mapping, the promotion plumbing or the undo stack shows up as a wrong count.
    """
    if depth == 0:
        return 1
    total = 0
    for move in game.legal_moves():
        undo = game.make(*move)
        total += _perft(game, depth - 1)
        game.unmake(undo)
    return total


def test_perft_matches_standard_chess_from_start():
    game = ChessGame()
    assert _perft(game, 1) == 20
    assert _perft(game, 2) == 400
    assert _perft(game, 3) == 8902


def test_promotion_choices_offer_the_four_pieces_queen_first():
    import chess

    game = ChessGame(board=chess.Board("r1b1k1nr/1P6/8/8/8/8/8/4K3 w kq - 0 1"))
    assert game.promotion_choices((1, 6), (1, 7)) == list(PROMOTION_PIECES)  # b7-b8
    assert game.promotion_choices((1, 6), (0, 7)) == list(PROMOTION_PIECES)  # bxa8
    assert game.promotion_choices((4, 0), (4, 1)) == []  # a king step
    assert game.promotion_choices((1, 6), (1, 5)) == []  # not a legal move at all


def test_perft_counts_promotions():
    """Perft from the start position never reaches a promotion — this one does.

    A pawn needs five moves to queen, so the shallow start-position counts above
    would pass with promotion entirely unimplemented. This position promotes on the
    first ply: a white pawn on b7 can push to b8 or take the rook on a8 or the bishop on c8, and
    each of the three lands as one of four pieces.
    """
    import chess

    game = ChessGame(board=chess.Board("r1b1k1nr/1P6/8/8/8/8/8/4K3 w kq - 0 1"))
    promotions = [m for m in game.legal_moves() if m.promotion is not None]
    assert len(promotions) == 12  # 3 destinations x 4 pieces
    assert {m.promotion for m in promotions} == set(PROMOTION_PIECES)

    # Known python-chess perft for this position, counted through our facade.
    assert _perft(game, 1) == len(game.legal_moves())
    assert _perft(game, 2) == _reference_perft(game.board, 2)
    assert _perft(game, 3) == _reference_perft(game.board, 3)


def _reference_perft(board, depth: int) -> int:
    """python-chess's own move stack, as an independent count."""
    if depth == 0:
        return 1
    total = 0
    for move in board.legal_moves:
        board.push(move)
        total += _reference_perft(board, depth - 1)
        board.pop()
    return total


# --------------------------------------------------------- control / attack map
def test_attacked_squares_slider_stops_at_blocker():
    from kaisparov.core.rules import attacked_squares

    # White rook on a1; nothing in the way -> controls the whole a-file and rank 1.
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.ROOK)
    controlled = attacked_squares(game.grid, Player.WHITE)
    assert (0, 7) in controlled  # far end of the file

    # Drop a blocker on a4: the rook attacks up to and including a4, nothing beyond.
    place(game, (0, 3), Player.WHITE, PieceType.PAWN)
    controlled = attacked_squares(game.grid, Player.WHITE)
    assert (0, 3) in controlled  # the blocker square itself is attacked
    assert (0, 4) not in controlled and (0, 7) not in controlled  # shadowed


def test_attacked_squares_pawn_controls_diagonals_not_push():
    from kaisparov.core.rules import attacked_squares

    game = empty_game()
    place(game, (3, 3), Player.WHITE, PieceType.PAWN)
    controlled = attacked_squares(game.grid, Player.WHITE)
    assert (2, 4) in controlled and (4, 4) in controlled  # forward diagonals
    assert (3, 4) not in controlled  # the push square is not an attack


def test_attacked_squares_matches_is_in_check_on_king_square():
    from kaisparov.core.rules import attacked_squares, is_in_check

    # Enemy rook checks the king along the file, unless blocked.
    game = empty_game()
    place(game, (0, 0), Player.WHITE, PieceType.KING)
    place(game, (0, 7), Player.BLACK, PieceType.ROOK)
    assert is_in_check(game.grid, Player.WHITE)
    assert (0, 0) in attacked_squares(game.grid, Player.BLACK)

    place(game, (0, 3), Player.WHITE, PieceType.PAWN)  # block the file
    assert not is_in_check(game.grid, Player.WHITE)
    assert (0, 0) not in attacked_squares(game.grid, Player.BLACK)


def test_is_in_check_agrees_with_the_control_map_on_random_positions():
    """Two independent routes to the same fact must never disagree.

    ``is_in_check`` asks python-chess whether one square is attacked;
    ``attacked_squares`` builds the whole control set from every piece's attack mask.
    They are different code paths through our wrappers, so scattering random cast
    lists across the board is a real cross-check of the coordinate mapping.
    """
    import random

    from kaisparov.core.rules import attacked_squares, find_king, is_in_check

    rng = random.Random(1234)
    # Exactly one king per side: standard chess has no other kind of position, and
    # python-chess picks a single king per colour when asked.
    types = [t for t in PieceType if t is not PieceType.KING]
    players = [Player.WHITE, Player.BLACK]
    squares = list(all_squares())

    for _ in range(400):
        game = empty_game()
        # Always give each side a king, then scatter a random cast of other pieces.
        spots = rng.sample(squares, rng.randint(2, 16))
        place(game, spots[0], Player.WHITE, PieceType.KING)
        place(game, spots[1], Player.BLACK, PieceType.KING)
        for coord in spots[2:]:
            place(game, coord, rng.choice(players), rng.choice(types))
        for player in players:
            enemy = Player.BLACK if player == Player.WHITE else Player.WHITE
            king = find_king(game.grid, player)
            assert is_in_check(game, player) == (king in attacked_squares(game, enemy))


def test_captured_by_sees_en_passant_without_playing_the_move():
    import chess

    game = ChessGame(board=chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"))
    before = game.fen()
    taken = game.captured_by((4, 4), (3, 5))  # exd6
    assert taken is not None and (taken.player, taken.type) == (Player.BLACK, PieceType.PAWN)
    assert game.captured_by((4, 4), (4, 5)) is None  # e5-e6
    assert game.fen() == before
    undo = game.make((4, 4), (3, 5))
    assert undo.captured is not None and undo.captured.type == PieceType.PAWN
    assert undo.captured_square == (3, 4)  # the pawn stood on d5, not d6
