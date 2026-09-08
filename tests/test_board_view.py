"""Board orientation: which side sits at the bottom, and when it is allowed to move.

The bug this pins: against an AI the view used to follow the side to move, so the
whole board turned over for the fraction of a second the AI spent thinking.
"""

from __future__ import annotations

import os

# GameInterface imports pygame; nothing here opens a window, but keep SDL headless.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from kaisparov.core.board import ChessGame  # noqa: E402
from kaisparov.core.game_interface import GameInterface, MatchSetup  # noqa: E402
from kaisparov.core.pieces import BOARD_SIZE, Player  # noqa: E402
from kaisparov.play import _board_orientation  # noqa: E402

W, B = Player.WHITE, Player.BLACK


def test_black_sees_the_board_upside_down():
    game = ChessGame()
    assert game.to_pov_coord((4, 1), W) == (4, 1)
    assert game.to_pov_coord((4, 1), B) == (4, BOARD_SIZE - 1 - 1)


def test_a_pinned_view_survives_a_change_of_turn():
    """The regression itself: the same square must stay where it was drawn."""
    game = ChessGame()
    ui = GameInterface(game)
    square = (4, 1)  # e2

    pinned = ui._to_display_coord(square, view_as=W)
    following = ui._to_display_coord(square, view_as=None)

    game.play((4, 1), (4, 3))  # White has moved; Black is now to move
    assert ui._to_display_coord(square, view_as=W) == pinned
    assert ui._to_display_coord(square, view_as=None) != following  # this is the flip


def test_display_and_real_coordinates_round_trip_for_either_side():
    game = ChessGame()
    ui = GameInterface(game)
    for view_as in (W, B, None):
        for square in ((0, 0), (4, 1), (7, 6)):
            display = ui._to_display_coord(square, view_as=view_as)
            assert ui._to_real_coord(display, view_as=view_as) == square


def test_only_two_humans_on_one_keyboard_get_a_flipping_board():
    assert _board_orientation(MatchSetup("vs_ai", W)) is W
    assert _board_orientation(MatchSetup("vs_ai", B)) is B
    assert _board_orientation(MatchSetup("ai_vs_ai", B)) is W  # spectator view is fixed
    assert _board_orientation(MatchSetup("solo", B)) is None
