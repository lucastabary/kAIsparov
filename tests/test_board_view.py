"""Board orientation: which side sits at the bottom, and when it may move.

The bug this pins: against an AI the view followed the side to move, so the whole
board turned over for the fraction of a second the AI spent thinking. The fix lives
in :class:`~kaisparov.core.board.ChessGame`'s viewpoint helpers, which now take the
player to render for instead of always using the side to move — so that is what is
tested here, with no pygame in sight.
"""

from __future__ import annotations

from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, Player

W, B = Player.WHITE, Player.BLACK
LAST = BOARD_SIZE - 1


def test_white_sees_the_board_as_it_is_and_black_sees_it_flipped():
    game = ChessGame()
    assert game.to_pov_coord((4, 1), W) == (4, 1)
    assert game.to_pov_coord((4, 1), B) == (4, LAST - 1)


def test_a_pinned_viewpoint_survives_a_change_of_turn():
    """The regression itself: a square must stay where it was drawn."""
    game = ChessGame()
    square = (4, 1)  # e2

    pinned = game.to_pov_coord(square, W)
    following = game.to_pov_coord(square)  # no player: follows the side to move

    game.play((4, 1), (4, 3))  # White has moved; Black is now to move
    assert game.to_pov_coord(square, W) == pinned
    assert game.to_pov_coord(square) != following  # this is the flip that flashed


def test_the_grid_is_pinned_too_not_just_the_coordinates():
    game = ChessGame()
    white_pawn = game.grid[4][1]

    game.play((4, 1), (4, 3))  # Black to move, but we render for White
    assert game.get_pov_grid(W)[4][3] is white_pawn
    assert game.get_pov_grid(B)[4][LAST - 3] is white_pawn
    assert game.get_pov_grid()[4][LAST - 3] is white_pawn  # follows the turn: Black


def test_the_viewpoint_transform_is_its_own_inverse():
    game = ChessGame()
    for player in (W, B, None):
        for square in ((0, 0), (4, 1), (7, 6)):
            shown = game.to_pov_coord(square, player)
            assert game.from_pov_coord(shown, player) == square


def test_omitting_the_player_means_the_side_to_move():
    game = ChessGame()
    assert game.to_pov_coord((4, 1)) == game.to_pov_coord((4, 1), W)
    game.play((4, 1), (4, 3))
    assert game.to_pov_coord((4, 1)) == game.to_pov_coord((4, 1), B)
