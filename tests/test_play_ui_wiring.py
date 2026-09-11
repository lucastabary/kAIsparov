"""The handful of decisions ``play.py`` makes on the interface's behalf.

Everything here reaches the pygame modules, which the rest of the suite avoids so it
runs anywhere (CI installs the project with ``--no-deps``). Nothing below opens a
window — but importing them still needs pygame present, so the whole module skips
when it is not.
"""

from __future__ import annotations

import os

import pytest

pygame = pytest.importorskip("pygame", reason="pygame UI modules are not installed")

# Nothing here opens a window; keep SDL headless anyway in case a driver is probed.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from kaisparov.core.board import ChessGame  # noqa: E402
from kaisparov.core.game_interface import GameInterface, MatchSetup  # noqa: E402
from kaisparov.core.notation import numbered_moves  # noqa: E402
from kaisparov.core.pieces import Player  # noqa: E402
from kaisparov.insights import MoveQuality  # noqa: E402
from kaisparov.play import _board_orientation, _legend_entries, _sidebar_layout  # noqa: E402

W, B = Player.WHITE, Player.BLACK


def test_only_two_humans_on_one_keyboard_get_a_flipping_board():
    assert _board_orientation(MatchSetup("vs_ai", W)) is W
    assert _board_orientation(MatchSetup("vs_ai", B)) is B
    assert _board_orientation(MatchSetup("ai_vs_ai", B)) is W  # spectator view is fixed
    assert _board_orientation(MatchSetup("solo", B)) is None


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
