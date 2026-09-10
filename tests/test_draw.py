"""Draw rules: repetition, no progress, insufficient material.

Torch-free and pygame-free: everything here runs on the pure-Python engine.
"""

from __future__ import annotations

import pytest

from kaisparov.core import draw
from kaisparov.core.board import ChessGame
from kaisparov.core.draw import DrawRules, is_insufficient_material
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player
from kaisparov.envs.chess_env import ChessEnv

W, B = Player.WHITE, Player.BLACK


def empty_grid():
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def place(grid, coord, player, piece_type):
    grid[coord[0]][coord[1]] = Piece(player, piece_type)


def shuffling_board():
    """Two kings and two rooks, each rook with an empty square to step onto."""
    grid = empty_grid()
    place(grid, (0, 0), W, PieceType.KING)
    place(grid, (7, 7), B, PieceType.KING)
    place(grid, (3, 0), W, PieceType.ROOK)
    place(grid, (3, 7), B, PieceType.ROOK)
    return grid


# A full there-and-back cycle for both rooks: the position after it is identical
# to the position before it (only the rooks' has_moved flag changes, on the first
# cycle only).
CYCLE = [((3, 0), (4, 0)), ((3, 7), (4, 7)), ((4, 0), (3, 0)), ((4, 7), (3, 7))]


# --------------------------------------------------------------- insufficient material


def test_bare_kings_are_insufficient():
    grid = empty_grid()
    place(grid, (0, 0), W, PieceType.KING)
    place(grid, (7, 7), B, PieceType.KING)
    assert is_insufficient_material(grid)


@pytest.mark.parametrize("minor", [PieceType.BISHOP, PieceType.KNIGHT])
def test_king_and_one_minor_is_insufficient(minor):
    grid = empty_grid()
    place(grid, (0, 0), W, PieceType.KING)
    place(grid, (7, 7), B, PieceType.KING)
    place(grid, (2, 2), W, minor)
    assert is_insufficient_material(grid)


def test_same_colour_bishops_are_insufficient_but_opposite_ones_are_not():
    def with_black_bishop_on(square):
        grid = empty_grid()
        place(grid, (0, 0), W, PieceType.KING)
        place(grid, (7, 7), B, PieceType.KING)
        place(grid, (2, 2), W, PieceType.BISHOP)  # light square: (2 + 2) % 2 == 0
        place(grid, square, B, PieceType.BISHOP)
        return grid

    assert is_insufficient_material(with_black_bishop_on((4, 4)))  # same complex
    assert not is_insufficient_material(with_black_bishop_on((4, 5)))  # other complex


@pytest.mark.parametrize(
    "extra",
    [PieceType.PAWN, PieceType.ROOK, PieceType.QUEEN],
)
def test_a_pawn_or_a_major_is_always_enough(extra):
    grid = empty_grid()
    place(grid, (0, 0), W, PieceType.KING)
    place(grid, (7, 7), B, PieceType.KING)
    place(grid, (2, 2), W, extra)
    assert not is_insufficient_material(grid)


def test_two_minors_are_enough():
    grid = empty_grid()
    place(grid, (0, 0), W, PieceType.KING)
    place(grid, (7, 7), B, PieceType.KING)
    place(grid, (2, 2), W, PieceType.KNIGHT)
    place(grid, (4, 2), W, PieceType.KNIGHT)
    assert not is_insufficient_material(grid)


# ------------------------------------------------------------------------ repetition


def test_threefold_repetition_is_a_draw():
    game = ChessGame(initial_board=shuffling_board())
    assert game.repetition_count() == 1
    assert game.draw_reason() is None

    # The first cycle flips both rooks' has_moved, so it lands on a *new* position;
    # from there each cycle repeats it. Three occurrences = threefold.
    for _ in range(3):
        for move in CYCLE:
            game.play(*move)

    assert game.repetition_count() == draw.REPETITION_LIMIT
    assert game.draw_reason() == draw.REPETITION


def test_a_different_position_does_not_count_as_a_repetition():
    game = ChessGame(initial_board=shuffling_board())
    for move in CYCLE:
        game.play(*move)
    game.play((3, 0), (5, 0))  # white rook goes somewhere else
    assert game.repetition_count() == 1


def test_repetition_ignores_positions_from_before_a_capture():
    """A capture makes earlier positions unreachable, so they can never repeat."""
    game = ChessGame(initial_board=shuffling_board())
    for move in CYCLE:
        game.play(*move)
    game.play((3, 0), (3, 7))  # rook takes rook
    assert game.repetition_count() == 1
    assert game.draw_reason() is None


# ----------------------------------------------------------------------- no progress


def test_no_progress_counter_resets_on_a_capture_or_a_pawn_move():
    grid = shuffling_board()
    place(grid, (0, 1), W, PieceType.PAWN)
    game = ChessGame(initial_board=grid)

    game.play((3, 0), (4, 0))
    game.play((3, 7), (4, 7))
    assert game.halfmove_clock == 2

    game.play((0, 1), (0, 2))  # pawn push
    assert game.halfmove_clock == 0

    game.play((4, 7), (4, 0))  # rook takes rook
    assert game.halfmove_clock == 0


def test_no_progress_ends_the_game():
    game = ChessGame(initial_board=shuffling_board())
    rules = DrawRules(repetition=0, no_progress_plies=4, insufficient_material=False)
    for move in CYCLE[:3]:
        game.play(*move)
        assert game.draw_reason(rules) is None
    game.play(*CYCLE[3])
    assert game.draw_reason(rules) == draw.NO_PROGRESS


# ------------------------------------------------------------------- state bookkeeping


def test_unmake_restores_the_draw_state():
    game = ChessGame(initial_board=shuffling_board())
    before = (game.zobrist, list(game.position_history), game.halfmove_clock)

    undo = game.make((3, 0), (4, 0))
    assert game.zobrist != before[0]
    game.unmake(undo)

    assert (game.zobrist, game.position_history, game.halfmove_clock) == before


def test_search_style_make_unmake_leaves_no_trace():
    """What a negamax does: many nested make/unmake pairs must be a no-op."""
    game = ChessGame(initial_board=shuffling_board())
    for move in CYCLE:
        game.play(*move)
    snapshot = (game.zobrist, list(game.position_history), game.halfmove_clock)

    undos = [game.make(*move) for move in CYCLE]
    for undo in reversed(undos):
        game.unmake(undo)

    assert (game.zobrist, game.position_history, game.halfmove_clock) == snapshot


def test_copy_carries_the_draw_state():
    game = ChessGame(initial_board=shuffling_board())
    for move in CYCLE:
        game.play(*move)
    clone = game.copy()

    assert clone.zobrist == game.zobrist
    assert clone.repetition_count() == game.repetition_count()
    assert clone.halfmove_clock == game.halfmove_clock

    clone.play(*CYCLE[0])  # the copy is independent
    assert clone.zobrist != game.zobrist


def test_the_same_position_hashes_the_same_however_it_was_reached():
    grid = shuffling_board()
    direct = ChessGame(initial_board=grid)
    for move in [((3, 0), (5, 0)), ((3, 7), (5, 7))]:
        direct.play(*move)

    detour = ChessGame(initial_board=grid)
    for move in [((3, 0), (4, 0)), ((3, 7), (4, 7)), ((4, 0), (5, 0)), ((4, 7), (5, 7))]:
        detour.play(*move)

    assert detour.zobrist == direct.zobrist


def test_castling_rights_are_part_of_the_fingerprint():
    """Every piece is back on its square, but the rooks have moved: a new position."""
    game = ChessGame(initial_board=shuffling_board())
    start = game.zobrist
    for move in CYCLE:
        game.play(*move)
    assert game.zobrist != start
    assert game.repetition_count() == 1


# -------------------------------------------------------------------------------- env


def test_env_scores_a_drawing_capture_on_material_only():
    grid = empty_grid()
    place(grid, (0, 0), W, PieceType.KING)
    place(grid, (7, 7), B, PieceType.KING)
    place(grid, (2, 2), W, PieceType.BISHOP)
    place(grid, (4, 4), B, PieceType.KNIGHT)

    env = ChessEnv()
    env.reset(board=grid)
    result = env.step(((2, 2), (4, 4)))  # bishop takes knight -> K+B vs K

    assert result.done
    assert env.winner is None
    assert env.end_reason == draw.INSUFFICIENT_MATERIAL
    assert result.reward == pytest.approx(3.0)  # the knight, and nothing for the draw


def test_env_scores_a_drawing_non_capture_at_zero():
    env = ChessEnv(draw_rules=DrawRules(repetition=2, no_progress_plies=0))
    env.reset(board=shuffling_board())

    result = None
    for _ in range(4):  # generous cap; the rooks repeat well before this
        for move in CYCLE:
            result = env.step(move)
            if result.done:
                break
        if result.done:
            break

    assert result is not None and result.done
    assert env.winner is None
    assert env.end_reason == draw.REPETITION
    assert result.reward == 0.0


def test_env_draw_rules_can_be_switched_off():
    grid = empty_grid()
    place(grid, (0, 0), W, PieceType.KING)
    place(grid, (7, 7), B, PieceType.KING)
    place(grid, (2, 2), W, PieceType.BISHOP)
    place(grid, (4, 4), B, PieceType.KNIGHT)

    env = ChessEnv(draw_rules=None)
    env.reset(board=grid)
    result = env.step(((2, 2), (4, 4)))
    assert not result.done
