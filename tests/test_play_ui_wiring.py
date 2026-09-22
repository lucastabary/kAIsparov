"""The handful of decisions ``play.py`` makes on the interface's behalf.

Everything here reaches the pygame modules, which the rest of the suite avoids so it
runs anywhere (CI installs the project with ``--no-deps``). Nothing below opens a
window — but importing them still needs pygame present, so the whole module skips
when it is not.
"""

from __future__ import annotations

import argparse
import os
import random

import pytest

pygame = pytest.importorskip("pygame", reason="pygame UI modules are not installed")

# Nothing here opens a window; keep SDL headless anyway in case a driver is probed.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from kaisparov.core.game import ChessGame  # noqa: E402
from kaisparov.core.game_interface import GameInterface, MatchSetup  # noqa: E402
from kaisparov.core.notation import numbered_moves  # noqa: E402
from kaisparov.core.pieces import Player  # noqa: E402
from kaisparov.insights import MoveQuality  # noqa: E402
from kaisparov.play import (  # noqa: E402
    _board_orientation,
    _draw_color,
    _game_over_message,
    _legend_entries,
    _setup_from_args,
    _sidebar_layout,
)

W, B = Player.WHITE, Player.BLACK


def test_only_two_humans_on_one_keyboard_get_a_flipping_board():
    assert _board_orientation(MatchSetup("vs_ai", W)) is W
    assert _board_orientation(MatchSetup("vs_ai", B)) is B
    assert _board_orientation(MatchSetup("ai_vs_ai", B)) is W  # spectator view is fixed
    assert _board_orientation(MatchSetup("solo", B)) is None


def test_a_random_colour_is_drawn_for_each_match_and_a_chosen_one_is_kept():
    rng = random.Random(0)
    drawn = {_draw_color(MatchSetup("vs_ai", None), rng).human_color for _ in range(40)}
    assert drawn == {W, B}
    assert _draw_color(MatchSetup("vs_ai", B), rng).human_color is B


def test_the_command_line_defaults_to_a_random_colour():
    args = argparse.Namespace(
        vs_ai=True, ai_vs_ai=False, solo=False, dev=False, review=False, color="random"
    )
    setup = _setup_from_args(args)
    assert setup is not None and setup.human_color is None


def test_the_menu_offers_random_between_white_and_black():
    layout = GameInterface()._menu_layout()
    white, random_, black = layout["white"], layout["random"], layout["black"]
    assert white.right < random_.left and random_.right < black.left
    assert white.y == random_.y == black.y


def test_the_legend_explains_every_grade_the_judge_can_hand_out():
    """A badge nobody can decode is a bug — the key must cover the whole enum."""
    entries = _legend_entries()
    assert [entry.tone for entry in entries] == [q.name.lower() for q in MoveQuality]
    assert [entry.symbol for entry in entries] == [q.symbol for q in MoveQuality]
    assert all(entry.title and entry.detail for entry in entries)


def test_every_legend_tone_has_a_colour():
    palette = GameInterface()._quality_colors
    assert all(entry.tone in palette for entry in _legend_entries())


def test_a_new_game_starts_with_a_blank_move_list():
    ui = GameInterface()
    ui.set_history(numbered_moves(["e4", "e5"]))
    ui._history_scroll = 3
    ui.set_game(ChessGame())
    assert ui.history == []
    assert ui._history_scroll == 0


def test_sidebar_buttons_sit_under_the_cards_not_inside_them():
    ui = GameInterface()
    ui.set_layout(_sidebar_layout(analysis=True, review=True, stepping=True))
    rects = ui._sidebar()
    cards = [rects["history"], rects["status"]]
    assert rects["history"].bottom < rects["status"].top
    for button in (rects["step"], rects["legend"]):
        assert not any(button.colliderect(card) for card in cards)
        assert button.top > rects["status"].bottom
    assert rects["step"].bottom < rects["legend"].top
    column = pygame.Rect(0, ui.margin, ui.window_width - ui.margin, ui.board_size_px)
    assert all(column.contains(rect) for rect in rects.values())


def test_a_plain_game_gives_the_move_list_the_whole_column():
    ui = GameInterface()
    ui.set_layout(_sidebar_layout(analysis=False, review=False, stepping=False))
    rects = ui._sidebar()
    assert set(rects) == {"history"}
    assert rects["history"].height == ui.board_size_px


def test_a_new_move_brings_the_list_back_to_the_latest_row():
    ui = GameInterface()
    ui._history_scroll = 5  # the reader had scrolled back
    ui.set_history(numbered_moves(["e4"]))
    assert ui._history_scroll == 0


def test_a_mated_side_ends_the_game_instead_of_waiting_for_a_move():
    # The game from the bug report: 7. Qxg6# leaves Black to move, and mated.
    game = ChessGame()
    for san in ["e4", "Nf6", "e5", "Ne4", "d3", "Nc5", "Qg4", "g6", "h4", "h5", "Qg5", "f6"]:
        game.board.push_san(san)
    assert _game_over_message(game) is None
    game.board.push_san("Qxg6")
    message = _game_over_message(game)
    assert message is not None and "mat" in message and "Blancs gagnent" in message


def test_a_stalemate_is_a_draw_on_the_game_over_panel():
    game = ChessGame()
    game.board.set_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    message = _game_over_message(game)
    assert message is not None and message.startswith("Partie nulle")


def test_the_promotion_picker_runs_from_the_last_rank_toward_the_middle():
    from kaisparov.core.game_interface import promotion_picker_cells

    assert promotion_picker_cells((1, 7)) == [(1, 7), (1, 6), (1, 5), (1, 4)]
    assert promotion_picker_cells((6, 0)) == [(6, 0), (6, 1), (6, 2), (6, 3)]


def _clicks(monkeypatch, ui: GameInterface, cells: list[tuple[int, int]]) -> None:
    """Feed left clicks on these display cells, one per event poll, as a human would.

    Posting them all at once would hand the whole batch to the move loop's first
    ``event.get()``, and the picker it opens would never see its click.
    """
    queue = [
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=ui._coord_to_rect(c).center)]
        for c in cells
    ]

    def get():
        assert queue, "the interface asked for more clicks than the test gave it"
        return queue.pop(0)

    monkeypatch.setattr(pygame.event, "get", get)


def test_a_human_promotion_plays_the_piece_picked(monkeypatch):
    import chess

    from kaisparov.core.move import Move
    from kaisparov.core.pieces import PieceType

    game = ChessGame(board=chess.Board("8/1P6/8/8/8/8/8/k3K3 w - - 0 1"))
    ui = GameInterface(game)
    ui._ensure_initialized()
    try:
        # The pawn on b7 to b8: the picker opens with the queen on b8 and the rook,
        # bishop and knight below it.
        _clicks(monkeypatch, ui, [(1, 6), (1, 7), (1, 4)])
        assert ui._get_single_move(view_as=W) == Move((1, 6), (1, 7), PieceType.KNIGHT)

        # A click off the picker takes the move back instead of queening.
        _clicks(monkeypatch, ui, [(1, 6), (1, 7), (6, 2), (4, 0), (4, 1)])
        assert ui._get_single_move(view_as=W) == Move((4, 0), (4, 1))
    finally:
        pygame.quit()
