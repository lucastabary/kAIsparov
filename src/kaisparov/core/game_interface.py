from __future__ import annotations

import math
from dataclasses import dataclass, replace

import pygame

from kaisparov.core.coords import Coord
from kaisparov.core.game import ChessGame
from kaisparov.core.move import PROMOTION_PIECES, Move
from kaisparov.core.notation import MoveRow
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player

Color = tuple[int, int, int]


@dataclass(frozen=True)
class ModelOption:
    """One selectable AI in the menu. ``key`` is opaque to the interface — the caller
    (``play.py``) maps it back to a policy (a run id, or a baseline like ``material``).
    """

    key: str
    label: str


@dataclass(frozen=True)
class MatchSetup:
    """The choice made on the pre-game menu (see :meth:`GameInterface.select_setup`)."""

    mode: str  # "solo" (two humans) | "vs_ai" (human vs model) | "ai_vs_ai"
    # Only meaningful for "vs_ai". ``None`` is the menu's "random": the caller draws a
    # side when the match starts (see ``play._draw_color``), so it is drawn anew for
    # every game rather than once for the whole session.
    human_color: Player | None = Player.WHITE
    dev_mode: bool = False  # surface the model's analysis while playing
    review_mode: bool = False  # grade each played move (chess.com-style badges)
    # Chosen model keys (see :class:`ModelOption`). ``ai_model`` is the opponent in
    # "vs_ai"; ``white_model``/``black_model`` are the two seats in "ai_vs_ai".
    ai_model: str | None = None
    white_model: str | None = None
    black_model: str | None = None


@dataclass(frozen=True)
class MoveArrow:
    """A rendering primitive for the developer overlay: a candidate move to draw.

    ``source``/``dest`` are engine coordinates (the interface flips them for POV);
    ``intensity`` in ``0..1`` scales the arrow's opacity and thickness so the
    strongest idea reads loudest. Kept free of any model type on purpose — the
    caller maps a :class:`~kaisparov.insights.PositionAnalysis` onto these.
    """

    source: Coord
    dest: Coord
    intensity: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class MoveBadge:
    """A rendering primitive for the move review: the grade of the move just played.

    Sits on the destination square, chess.com style. ``tone`` selects a colour from
    the interface palette (see ``_colors["quality"]``) and is the *name* of a
    :class:`~kaisparov.insights.MoveQuality`; the interface deliberately does not
    import that enum, so the engine package keeps knowing nothing about analysis.
    """

    square: Coord
    symbol: str  # two characters at most, e.g. "!!" or "??"
    tone: str = "good"
    caption: str = ""  # spelled-out grade for the side panel


@dataclass(frozen=True)
class LegendEntry:
    """One row of the "what do these badges mean?" panel.

    A badge is two characters wide, so the key to reading it has to live somewhere.
    The caller owns the wording (and the order — best first); the interface only
    knows how to lay the rows out. Assign a list of these to
    :attr:`GameInterface.legend` and the button appears on its own.
    """

    symbol: str
    title: str  # the grade, spelled out
    detail: str  # one line on what earns it
    tone: str = "good"


@dataclass(frozen=True)
class SidebarLayout:
    """What the column beside the board makes room for, for the length of a match.

    Decided once, up front, so nothing in the column changes size between frames:
    the move list keeps its height while the AI thinks, and the buttons stay put.
    The buttons sit under the cards, never inside them.
    """

    status_rows: int = 0  # lines of the analysis card under the move list (0: no card)
    step_button: bool = False  # "Coup suivant" — AI vs AI, one move at a time
    legend_button: bool = False  # the key to the review badges


def promotion_picker_cells(dest_display: tuple[int, int]) -> list[tuple[int, int]]:
    """Display cells of the promotion picker, one per piece of ``PROMOTION_PIECES``.

    A column that starts on the promotion square and runs toward the middle of the
    board, as on chess.com: down from the top edge, up from the bottom one (a board
    seen from Black's side puts White's last rank at the bottom).
    """
    x, y = dest_display
    step = -1 if y == BOARD_SIZE - 1 else 1
    return [(x, y + step * i) for i in range(len(PROMOTION_PIECES))]


class GameInterface:
    """Graphical interface for rendering a ChessGame in a separate window."""

    _SIDEBAR_GAP = 12  # between two blocks of the column, and from the board
    _BUTTON_H = 48
    _STATUS_ROW_H = 26

    def __init__(self, game: ChessGame | None = None):
        self.game: ChessGame | None = game

        self.cell_size = 88
        self.board_size_px = self.cell_size * BOARD_SIZE
        self.margin = 34
        self.sidebar_width = 290
        self.window_width = self.board_size_px + self.margin * 2 + self.sidebar_width
        self.window_height = self.board_size_px + self.margin * 2

        # Rows for the badge legend, best grade first. Empty means "no review in
        # this session", and the button that opens it stays hidden.
        self.legend: list[LegendEntry] = []

        # What the column beside the board holds this match (see set_layout).
        self.layout = SidebarLayout()

        # The scoresheet of the game on the board (see set_history), and how many
        # rows the reader has scrolled back from the latest move.
        self.history: list[MoveRow] = []
        self._history_scroll = 0

        self._initialized = False
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._piece_sprites: dict[tuple[Player, PieceType], pygame.Surface] = {}
        self._highlight_surface: pygame.Surface | None = None

        self._colors: dict[str, Color] = {
            "bg_start": (18, 24, 38),
            "bg_end": (32, 45, 70),
            "board_light": (233, 220, 198),
            "board_dark": (128, 92, 66),
            "board_border": (20, 22, 30),
            "panel": (19, 29, 45),
            "panel_text": (240, 243, 250),
            "panel_subtext": (168, 181, 207),
            "accent": (89, 173, 255),
            "button": (30, 44, 68),
            "button_hover": (44, 66, 102),
            "button_border": (89, 173, 255),
            "button_active": (46, 84, 140),
            "analysis": (89, 173, 255),  # developer-overlay arrows/tints
        }

        # Move-review badges, keyed by MoveQuality name (see MoveBadge.tone).
        self._quality_colors: dict[str, Color] = {
            "brilliant": (38, 194, 168),
            "great": (90, 155, 214),
            "best": (129, 182, 76),
            "excellent": (150, 191, 91),
            "good": (150, 166, 140),
            "forced": (140, 148, 166),
            "inaccuracy": (231, 183, 51),
            "mistake": (233, 145, 71),
            "miss": (219, 105, 96),
            "blunder": (219, 66, 47),
        }

    def set_game(self, game: ChessGame) -> None:
        """Assigns a ChessGame instance to this interface (with a blank history)."""
        self.game = game
        self.set_history([])

    def set_history(self, rows: list[MoveRow]) -> None:
        """Replace the move list, and jump back to its latest move.

        The caller owns the notation (see :mod:`kaisparov.core.notation`); the
        interface only lays the rows out. A new move always brings the list back to
        the bottom, so scrolling back never hides the move just played.
        """
        self.history = rows
        self._history_scroll = 0

    def set_layout(self, layout: SidebarLayout) -> None:
        """Say what the column beside the board holds for the match about to start."""
        self.layout = layout

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        pygame.init()
        pygame.display.set_caption("kAIsparov - Chess Interface")
        self._screen = pygame.display.set_mode((self.window_width, self.window_height))
        self._clock = pygame.time.Clock()
        self._build_piece_sprites()
        # Translucent tint for the last move's from/to squares (chess.com-style).
        self._highlight_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        self._highlight_surface.fill((245, 214, 71, 90))
        self._initialized = True

    def _make_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        candidates = ["segoeuiemoji", "segoe ui symbol", "arial unicode ms", "dejavusans"]
        for name in candidates:
            font = pygame.font.SysFont(name, size, bold=bold)
            if font is not None:
                return font
        return pygame.font.SysFont(None, size, bold=bold)

    def _make_piece_font(self, size: int) -> pygame.font.Font:
        # Fonts that carry monochrome chess glyphs (U+2654–265F), best first.
        for name in ("Segoe UI Symbol", "DejaVu Sans", "Arial Unicode MS", "FreeSerif"):
            path = pygame.font.match_font(name)
            if path:
                return pygame.font.Font(path, size)
        return pygame.font.SysFont(None, size)

    def _build_piece_sprites(self) -> None:
        # Solid chess figures (drawn, not letters), colored per side with an outline.
        glyphs = {
            PieceType.KING: "♚",
            PieceType.QUEEN: "♛",
            PieceType.ROOK: "♜",
            PieceType.BISHOP: "♝",
            PieceType.KNIGHT: "♞",
            PieceType.PAWN: "♟",
        }
        font = self._make_piece_font(int(self.cell_size * 0.74))
        outline_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2), (-2, 2), (2, -2)]

        for player in (Player.WHITE, Player.BLACK):
            if player == Player.WHITE:
                fill, outline = (248, 249, 252), (24, 28, 38)
            else:
                fill, outline = (30, 34, 46), (232, 236, 244)

            for piece_type, glyph in glyphs.items():
                surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                center = (self.cell_size // 2, self.cell_size // 2)

                outline_img = font.render(glyph, True, outline)
                for dx, dy in outline_offsets:
                    surface.blit(
                        outline_img, outline_img.get_rect(center=(center[0] + dx, center[1] + dy))
                    )

                fill_img = font.render(glyph, True, fill)
                surface.blit(fill_img, fill_img.get_rect(center=center))

                self._piece_sprites[(player, piece_type)] = surface

    def _last_move_display_cells(self, view_as: Player | None) -> set[tuple[int, int]]:
        """Display coords of the last move's from/to squares (empty if no move yet)."""
        if self.game is None or self.game.last_move is None:
            return set()
        return {self._to_display_coord(square, view_as=view_as) for square in self.game.last_move}

    def _draw_gradient_background(self) -> None:
        assert self._screen is not None
        for y in range(self.window_height):
            t = y / max(1, self.window_height - 1)
            r = int(self._colors["bg_start"][0] * (1 - t) + self._colors["bg_end"][0] * t)
            g = int(self._colors["bg_start"][1] * (1 - t) + self._colors["bg_end"][1] * t)
            b = int(self._colors["bg_start"][2] * (1 - t) + self._colors["bg_end"][2] * t)
            pygame.draw.line(self._screen, (r, g, b), (0, y), (self.window_width, y))

    def _draw_board(self, view_as: Player | None) -> None:
        assert self._screen is not None
        assert self.game is not None

        board_left = self.margin
        board_top = self.margin

        grid = self.game.get_pov_grid(self._viewer(view_as))

        board_rect = pygame.Rect(
            board_left - 4, board_top - 4, self.board_size_px + 8, self.board_size_px + 8
        )
        pygame.draw.rect(self._screen, self._colors["board_border"], board_rect, border_radius=12)

        last_move_cells = self._last_move_display_cells(view_as)

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                rect = pygame.Rect(
                    board_left + x * self.cell_size,
                    board_top + (BOARD_SIZE - 1 - y) * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                is_light = (x + y) % 2 == 0
                cell_color = self._colors["board_light"] if is_light else self._colors["board_dark"]
                pygame.draw.rect(self._screen, cell_color, rect)

                if (x, y) in last_move_cells and self._highlight_surface is not None:
                    self._screen.blit(self._highlight_surface, rect.topleft)

                piece = grid[x][y]
                if piece is not None:
                    sprite = self._piece_sprites.get((piece.player, piece.type))
                    if sprite is not None:
                        sprite_rect = sprite.get_rect(center=rect.center)
                        self._screen.blit(sprite, sprite_rect)

    def _draw_selection_overlay(
        self,
        selected_coord: tuple[int, int] | None,
        possible_destinations: set[tuple[int, int]],
        view_as: Player | None,
    ) -> None:
        assert self._screen is not None

        if selected_coord is not None:
            x, y = self._to_display_coord(selected_coord, view_as=view_as)
            rect = self._coord_to_rect((x, y))
            pygame.draw.rect(self._screen, (245, 214, 71), rect, width=5, border_radius=9)

        for dest in possible_destinations:
            x, y = self._to_display_coord(dest, view_as=view_as)
            rect = self._coord_to_rect((x, y))
            pygame.draw.circle(self._screen, (82, 196, 26), rect.center, 11)
            pygame.draw.circle(self._screen, (242, 255, 233), rect.center, 11, width=2)

    def _draw_analysis_overlay(self, arrows: list[MoveArrow], view_as: Player | None) -> None:
        """Developer mode: draw the model's candidate moves as tinted arrows.

        Everything is painted on a translucent surface so weaker ideas fade back
        and the board stays readable. The strongest arrow is drawn last (on top).
        """
        assert self._screen is not None
        if not arrows:
            return

        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        label_font = self._make_font(17, bold=True)

        for arrow in sorted(arrows, key=lambda a: a.intensity):
            intensity = max(0.0, min(1.0, arrow.intensity))
            src_disp = self._to_display_coord(arrow.source, view_as=view_as)
            dst_disp = self._to_display_coord(arrow.dest, view_as=view_as)
            src_rect = self._coord_to_rect(src_disp)
            dst_rect = self._coord_to_rect(dst_disp)

            r, g, b = self._colors["analysis"]
            alpha = int(70 + 150 * intensity)
            width = int(4 + 7 * intensity)

            # Faint tint on the destination square so the target reads even at a glance.
            tint = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            tint.fill((r, g, b, int(28 + 60 * intensity)))
            overlay.blit(tint, dst_rect.topleft)

            self._draw_arrow(overlay, src_rect.center, dst_rect.center, (r, g, b, alpha), width)

            if arrow.label:
                chip = label_font.render(arrow.label, True, (240, 246, 255))
                chip_bg = chip.get_rect()
                chip_bg.center = dst_rect.center
                chip_bg.inflate_ip(10, 6)
                pygame.draw.rect(overlay, (12, 18, 30, 210), chip_bg, border_radius=6)
                overlay.blit(chip, chip.get_rect(center=dst_rect.center))

        self._screen.blit(overlay, (0, 0))

    def _draw_move_badge(self, badge: MoveBadge | None, view_as: Player | None) -> None:
        """Draw the grade of the last move on its destination square.

        A coloured disc pinned to the square's top-right corner, the way chess.com
        marks up a reviewed game — close enough to the move to read at a glance,
        small enough not to hide the piece underneath.
        """
        assert self._screen is not None
        if badge is None:
            return

        rect = self._coord_to_rect(self._to_display_coord(badge.square, view_as=view_as))
        radius = 18
        centre = (rect.right - radius + 4, rect.top + radius - 4)
        self._draw_badge_disc(centre, badge.symbol, badge.tone, radius=radius)

    def _draw_badge_disc(
        self, centre: tuple[int, int], symbol: str, tone: str, radius: int = 18
    ) -> None:
        """The badge itself — shared by the board and the legend so they cannot drift."""
        assert self._screen is not None
        color = self._quality_colors.get(tone, self._colors["accent"])
        pygame.draw.circle(self._screen, color, centre, radius)
        pygame.draw.circle(self._screen, (250, 252, 255), centre, radius, width=3)

        font = self._make_font(radius + 2 if len(symbol) < 2 else radius - 1, bold=True)
        text = font.render(symbol, True, (18, 24, 34))
        self._screen.blit(text, text.get_rect(center=centre))

    def _draw_arrow(
        self,
        surface: pygame.Surface,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int, int],
        width: int,
    ) -> None:
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        head = 12 + width
        # Stop the shaft short of the tip so the head sits cleanly on the target.
        shaft_end = (end[0] - head * 0.7 * math.cos(angle), end[1] - head * 0.7 * math.sin(angle))
        pygame.draw.line(surface, color, start, shaft_end, width)

        left = (
            end[0] - head * math.cos(angle - math.radians(28)),
            end[1] - head * math.sin(angle - math.radians(28)),
        )
        right = (
            end[0] - head * math.cos(angle + math.radians(28)),
            end[1] - head * math.sin(angle + math.radians(28)),
        )
        pygame.draw.polygon(surface, color, [end, left, right])

    def _coord_to_rect(self, display_coord: tuple[int, int]) -> pygame.Rect:
        x, y = display_coord
        board_left = self.margin
        board_top = self.margin
        return pygame.Rect(
            board_left + x * self.cell_size,
            board_top + (BOARD_SIZE - 1 - y) * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

    def _pixel_to_display_coord(self, mouse_pos: tuple[int, int]) -> tuple[int, int] | None:
        px, py = mouse_pos
        board_left = self.margin
        board_top = self.margin

        if px < board_left or py < board_top:
            return None

        rel_x = px - board_left
        rel_y = py - board_top
        if rel_x >= self.board_size_px or rel_y >= self.board_size_px:
            return None

        x = rel_x // self.cell_size
        y_from_top = rel_y // self.cell_size
        y = BOARD_SIZE - 1 - y_from_top
        return (int(x), int(y))

    def _viewer(self, view_as: Player | None) -> Player:
        """Whose side of the board is at the bottom.

        ``None`` means "follow the side to move", which flips the board every ply —
        right for two humans sharing a keyboard, wrong for everything else, since a
        board that turns over while the opponent thinks is just disorienting.
        """
        assert self.game is not None
        return self.game.turn if view_as is None else view_as

    def _to_real_coord(
        self, display_coord: tuple[int, int], view_as: Player | None
    ) -> tuple[int, int]:
        assert self.game is not None
        return self.game.from_pov_coord(display_coord, self._viewer(view_as))

    def _to_display_coord(
        self, real_coord: tuple[int, int], view_as: Player | None
    ) -> tuple[int, int]:
        assert self.game is not None
        return self.game.to_pov_coord(real_coord, self._viewer(view_as))

    # ---------------------------------------------------------------- sidebar
    def _sidebar(self) -> dict[str, pygame.Rect]:
        """Where each block of the column beside the board sits, stacked bottom-up.

        The buttons hug the bottom edge, outside any card; the analysis card, when
        the match reserved one, sits above them; the move list takes whatever height
        is left. Keys: ``history`` always, ``status``/``step``/``legend`` when the
        layout asks for them.
        """
        layout = self.layout
        gap = self._SIDEBAR_GAP
        left = self.margin + self.board_size_px + gap * 2
        width = self.window_width - self.margin - left
        bottom = self.margin + self.board_size_px

        rects: dict[str, pygame.Rect] = {}
        for name, wanted in (("legend", layout.legend_button), ("step", layout.step_button)):
            if wanted:
                rects[name] = pygame.Rect(left, bottom - self._BUTTON_H, width, self._BUTTON_H)
                bottom -= self._BUTTON_H + gap
        if layout.status_rows > 0:
            height = 20 + layout.status_rows * self._STATUS_ROW_H
            rects["status"] = pygame.Rect(left, bottom - height, width, height)
            bottom -= height + gap
        rects["history"] = pygame.Rect(left, self.margin, width, bottom - self.margin)
        return rects

    def _draw_sidebar(self, status_lines: list[str] | None) -> None:
        assert self._screen is not None
        rects = self._sidebar()
        self._draw_history_panel(rects["history"])

        if "status" in rects:
            rect = rects["status"]
            self._draw_card(rect)
            font = self._make_font(18, bold=False)
            for i, line in enumerate((status_lines or [])[: self.layout.status_rows]):
                text = font.render(line, True, self._colors["panel_subtext"])
                y = rect.y + 10 + i * self._STATUS_ROW_H
                self._screen.blit(
                    text, (rect.x + 16, y + (self._STATUS_ROW_H - text.get_height()) // 2)
                )

        if "legend" in rects and self.legend:
            rect = rects["legend"]
            self._draw_button(
                rect,
                "Legende des notes  (L)",
                hover=rect.collidepoint(pygame.mouse.get_pos()),
                font_size=20,
            )

    def _draw_card(self, rect: pygame.Rect) -> None:
        assert self._screen is not None
        pygame.draw.rect(self._screen, self._colors["panel"], rect, border_radius=16)
        pygame.draw.rect(self._screen, self._colors["accent"], rect, width=2, border_radius=16)

    # ----------------------------------------------------------------- legend
    def _legend_hit(self, pos: tuple[int, int]) -> bool:
        rect = self._sidebar().get("legend")
        return bool(self.legend) and rect is not None and rect.collidepoint(pos)

    def show_legend(self, over_board: bool = True, view_as: Player | None = None) -> bool:
        """Show what each grade badge means, until dismissed.

        Reachable both before a game (from the menu) and during one (the button
        under the move list, or the ``L`` key), because the two characters on a badge only make
        sense once — after that the reader wants the board back.

        Returns ``False`` if the window was closed, ``True`` when dismissed.
        """
        self._ensure_initialized()
        assert self._screen is not None
        assert self._clock is not None
        if not self.legend:
            return True

        cx, cy = self.window_width // 2, self.window_height // 2
        row_h = 44
        height = 104 + row_h * len(self.legend) + 136
        panel = pygame.Rect(cx - 390, cy - height // 2, 780, height)
        close_btn = pygame.Rect(cx - 110, panel.bottom - 74, 220, 50)

        title_font = self._make_font(36, bold=True)
        name_font = self._make_font(21, bold=True)
        detail_font = self._make_font(18, bold=False)
        foot_font = self._make_font(17, bold=False)
        close_keys = (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE, pygame.K_l)

        def dismisses(event) -> bool:
            """A left click on the close button — or anywhere off the panel."""
            if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
                return False
            return close_btn.collidepoint(event.pos) or not panel.collidepoint(event.pos)

        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key in close_keys:
                    return True
                if dismisses(event):
                    return True

            if over_board and self.game is not None:
                self._draw_frame(view_as=view_as)
            else:
                self._draw_gradient_background()
            dim = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
            dim.fill((10, 14, 24, 200))
            self._screen.blit(dim, (0, 0))

            pygame.draw.rect(self._screen, self._colors["panel"], panel, border_radius=18)
            pygame.draw.rect(self._screen, self._colors["accent"], panel, width=2, border_radius=18)
            title = title_font.render("Notes des coups", True, self._colors["panel_text"])
            self._screen.blit(title, title.get_rect(center=(cx, panel.y + 46)))

            for i, entry in enumerate(self.legend):
                y = panel.y + 96 + i * row_h + row_h // 2
                self._draw_badge_disc((panel.x + 52, y), entry.symbol, entry.tone, radius=17)
                name = name_font.render(entry.title, True, self._colors["panel_text"])
                self._screen.blit(name, (panel.x + 86, y - name.get_height() // 2))
                detail = detail_font.render(entry.detail, True, self._colors["panel_subtext"])
                self._screen.blit(detail, (panel.x + 324, y - detail.get_height() // 2))

            foot = foot_font.render(
                "Les 'pts' sont des points de probabilite de gain perdus face au meilleur coup.",
                True,
                self._colors["panel_subtext"],
            )
            self._screen.blit(foot, foot.get_rect(center=(cx, panel.bottom - 120)))
            self._draw_button(
                close_btn, "Fermer", hover=close_btn.collidepoint(mouse), font_size=21
            )

            pygame.display.flip()
            self._clock.tick(60)

    # ---------------------------------------------------------------- history
    _HISTORY_TOP = 78  # title band above the first row
    _HISTORY_ROW_H = 26

    def _history_rect(self) -> pygame.Rect:
        """The move-list card, at the top of the column beside the board."""
        return self._sidebar()["history"]

    def _history_capacity(self) -> int:
        """How many rows fit below the title."""
        return (self._history_rect().height - self._HISTORY_TOP - 12) // self._HISTORY_ROW_H

    def _scroll_history(self, event) -> None:
        """Mouse wheel over the move list: scroll back through the game (or forward)."""
        if event.type != pygame.MOUSEWHEEL:
            return
        if not self._history_rect().collidepoint(pygame.mouse.get_pos()):
            return
        max_scroll = max(0, len(self.history) - self._history_capacity())
        self._history_scroll = max(0, min(max_scroll, self._history_scroll + event.y))

    def _draw_history_panel(self, rect: pygame.Rect) -> None:
        """The scoresheet: one row per move number, White then Black.

        The latest rows are shown by default; the wheel scrolls back, and the move
        just played sits on an accent chip so it reads at a glance.
        """
        assert self._screen is not None
        self._draw_card(rect)

        title_font = self._make_font(26, bold=True)
        hint_font = self._make_font(15, bold=False)
        move_font = self._make_font(19, bold=False)
        latest_font = self._make_font(19, bold=True)

        title = title_font.render("Coups", True, self._colors["panel_text"])
        self._screen.blit(title, (rect.x + 16, rect.y + 16))

        rows = self.history
        if not rows:
            empty = hint_font.render("Aucun coup joue.", True, self._colors["panel_subtext"])
            self._screen.blit(empty, (rect.x + 16, rect.y + self._HISTORY_TOP))
            return

        capacity = self._history_capacity()
        last = max(0, len(rows) - self._history_scroll)
        first = max(0, last - capacity)
        if len(rows) > capacity:
            hint = "molette: defiler" if self._history_scroll == 0 else "molette: revenir"
            text = hint_font.render(hint, True, self._colors["panel_subtext"])
            self._screen.blit(text, (rect.x + 16, rect.y + 54))

        # The move just played: Black's reply on the last row, or White's move if
        # Black has not answered yet.
        latest = (len(rows) - 1, "black" if rows[-1].black else "white")
        number_x, white_x, black_x = rect.x + 16, rect.x + 64, rect.x + 150

        for offset, index in enumerate(range(first, last)):
            row = rows[index]
            y = rect.y + self._HISTORY_TOP + offset * self._HISTORY_ROW_H
            band = pygame.Rect(rect.x + 6, y, rect.width - 12, self._HISTORY_ROW_H)
            if index % 2:
                pygame.draw.rect(self._screen, self._colors["button"], band, border_radius=6)

            number = move_font.render(f"{row.number}.", True, self._colors["panel_subtext"])
            self._screen.blit(number, (number_x, band.centery - number.get_height() // 2))

            for side, x, san in (("white", white_x, row.white), ("black", black_x, row.black)):
                if not san:
                    continue
                is_latest = (index, side) == latest
                font = latest_font if is_latest else move_font
                text = font.render(san, True, self._colors["panel_text"])
                if is_latest:
                    chip = text.get_rect(topleft=(x, band.centery - text.get_height() // 2))
                    pygame.draw.rect(
                        self._screen,
                        self._colors["button_active"],
                        chip.inflate(10, 4),
                        border_radius=5,
                    )
                self._screen.blit(text, (x, band.centery - text.get_height() // 2))

    def _draw_frame(
        self,
        view_as: Player | None,
        selected_coord: tuple[int, int] | None = None,
        possible_destinations: set[tuple[int, int]] | None = None,
        analysis_arrows: list[MoveArrow] | None = None,
        status_lines: list[str] | None = None,
        badge: MoveBadge | None = None,
    ) -> None:
        self._draw_gradient_background()
        self._draw_board(view_as=view_as)
        self._draw_analysis_overlay(analysis_arrows or [], view_as=view_as)
        self._draw_selection_overlay(
            selected_coord=selected_coord,
            possible_destinations=possible_destinations or set(),
            view_as=view_as,
        )
        self._draw_move_badge(badge, view_as=view_as)
        self._draw_sidebar(status_lines)

    def render(self, view_as: Player | None = None, fps: int = 60) -> None:
        """Opens a dedicated window and renders the game continuously.

        Args:
                view_as: Whose side sits at the bottom; None follows the side to move.
                fps: Max refresh rate.
        """
        if self.game is None:
            raise ValueError("No ChessGame assigned. Use set_game(...) first.")

        self._ensure_initialized()
        assert self._screen is not None
        assert self._clock is not None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self._draw_frame(view_as=view_as)
            pygame.display.flip()
            self._clock.tick(fps)

        pygame.quit()
        self._initialized = False

    # ------------------------------------------------------------------- menu
    def _menu_layout(self) -> dict[str, pygame.Rect]:
        """Clickable regions of the start menu, centred over the window."""
        cx = self.window_width // 2
        width = 460
        left = cx - width // 2
        button_h = 64
        third = (width - 2 * 12) // 3
        return {
            "solo": pygame.Rect(left, 232, width, button_h),
            "vs_ai": pygame.Rect(left, 312, width, button_h),
            "ai_vs_ai": pygame.Rect(left, 392, width, button_h),
            "white": pygame.Rect(left, 512, third, 48),
            "random": pygame.Rect(left + third + 12, 512, third, 48),
            "black": pygame.Rect(left + 2 * (third + 12), 512, third, 48),
            "dev": pygame.Rect(left, 578, width, 40),
            "review": pygame.Rect(left, 620, width, 40),
            "legend": pygame.Rect(left, 668, width, 42),
        }

    def _draw_button(
        self,
        rect: pygame.Rect,
        label: str,
        hover: bool = False,
        active: bool = False,
        font_size: int = 26,
    ) -> None:
        assert self._screen is not None
        if active:
            bg = self._colors["button_active"]
        elif hover:
            bg = self._colors["button_hover"]
        else:
            bg = self._colors["button"]
        border = self._colors["accent"] if (active or hover) else self._colors["button_border"]
        pygame.draw.rect(self._screen, bg, rect, border_radius=12)
        pygame.draw.rect(self._screen, border, rect, width=2, border_radius=12)
        font = self._make_font(font_size, bold=True)
        text = font.render(label, True, self._colors["panel_text"])
        self._screen.blit(text, text.get_rect(center=rect.center))

    def _draw_checkbox(
        self, rect: pygame.Rect, label: str, checked: bool, hover: bool = False
    ) -> None:
        assert self._screen is not None
        box = pygame.Rect(rect.x, rect.y + (rect.height - 26) // 2, 26, 26)
        pygame.draw.rect(self._screen, self._colors["button"], box, border_radius=6)
        border = self._colors["accent"] if (checked or hover) else self._colors["button_border"]
        pygame.draw.rect(self._screen, border, box, width=2, border_radius=6)
        if checked:
            pygame.draw.rect(
                self._screen, self._colors["accent"], box.inflate(-10, -10), border_radius=3
            )
        font = self._make_font(22, bold=True)
        text = font.render(label, True, self._colors["panel_text"])
        self._screen.blit(text, (box.right + 14, box.centery - text.get_height() // 2))

    def select_setup(self, models: list[ModelOption] | None = None) -> MatchSetup | None:
        """Show the pre-game menu and return the chosen setup (``None`` if closed).

        Pick a mode (solo, vs AI, AI vs AI); the colour toggle (used only vs the AI)
        and the two switches (developer mode, move review) persist until a mode is
        clicked. Solo starts immediately; the AI modes open a model-selection screen
        first so the user chooses which trained model plays each AI seat.

        The colour starts on "random", which leaves ``human_color`` as ``None`` for
        the caller to draw.
        """
        self._ensure_initialized()
        assert self._screen is not None
        assert self._clock is not None

        models = models or []
        color: Player | None = None
        dev = False
        review = False
        cx = self.window_width // 2

        title_font = self._make_font(66, bold=True)
        subtitle_font = self._make_font(24, bold=False)
        label_font = self._make_font(20, bold=True)
        hint_font = self._make_font(18, bold=False)

        while True:
            layout = self._menu_layout()
            mouse = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if layout["solo"].collidepoint(pos):
                        return MatchSetup("solo", color, dev, review)
                    if layout["vs_ai"].collidepoint(pos):
                        setup = self._select_models("vs_ai", color, dev, review, models)
                        if setup != "back":
                            return setup
                    if layout["ai_vs_ai"].collidepoint(pos):
                        setup = self._select_models("ai_vs_ai", color, dev, review, models)
                        if setup != "back":
                            return setup
                    if layout["white"].collidepoint(pos):
                        color = Player.WHITE
                    if layout["random"].collidepoint(pos):
                        color = None
                    if layout["black"].collidepoint(pos):
                        color = Player.BLACK
                    if layout["dev"].collidepoint(pos):
                        dev = not dev
                    if layout["review"].collidepoint(pos):
                        review = not review
                    if self.legend and layout["legend"].collidepoint(pos):
                        opened = self.show_legend(over_board=False)
                        if not opened:
                            return None

            self._draw_gradient_background()

            title = title_font.render("kAIsparov", True, self._colors["panel_text"])
            self._screen.blit(title, title.get_rect(center=(cx, 118)))
            subtitle = subtitle_font.render(
                "Choisissez un mode de jeu", True, self._colors["panel_subtext"]
            )
            self._screen.blit(subtitle, subtitle.get_rect(center=(cx, 178)))

            self._draw_button(
                layout["solo"], "Jouer seul  (2 joueurs)", hover=layout["solo"].collidepoint(mouse)
            )
            self._draw_button(
                layout["vs_ai"], "Jouer contre une IA", hover=layout["vs_ai"].collidepoint(mouse)
            )
            self._draw_button(
                layout["ai_vs_ai"], "IA contre IA", hover=layout["ai_vs_ai"].collidepoint(mouse)
            )

            color_label = label_font.render(
                "Votre couleur (contre l'IA)", True, self._colors["panel_subtext"]
            )
            self._screen.blit(color_label, (layout["white"].x, layout["white"].y - 30))
            for key, label, choice in (
                ("white", "Blancs", Player.WHITE),
                ("random", "Aleatoire", None),
                ("black", "Noirs", Player.BLACK),
            ):
                self._draw_button(
                    layout[key],
                    label,
                    hover=layout[key].collidepoint(mouse),
                    active=color == choice,
                    font_size=22,
                )

            self._draw_checkbox(
                layout["dev"],
                "Mode developpeur (analyse du modele)",
                dev,
                hover=layout["dev"].collidepoint(mouse),
            )
            self._draw_checkbox(
                layout["review"],
                "Analyse des coups (notes style chess.com)",
                review,
                hover=layout["review"].collidepoint(mouse),
            )
            if self.legend:
                self._draw_button(
                    layout["legend"],
                    "Que veulent dire les notes ?",
                    hover=layout["legend"].collidepoint(mouse),
                    font_size=20,
                )

            hint = hint_font.render(
                "Cliquez un mode pour commencer.",
                True,
                self._colors["panel_subtext"],
            )
            self._screen.blit(hint, hint.get_rect(center=(cx, 736)))

            pygame.display.flip()
            self._clock.tick(60)

    # --------------------------------------------------------- model selection
    def _model_row_rects(
        self, area: pygame.Rect, count: int, row_h: int, scroll: int
    ) -> list[tuple[int, pygame.Rect]]:
        """Rects for every model row in ``area`` (some may fall outside — the caller
        clips drawing and hit-testing to ``area``)."""
        rects = []
        for i in range(count):
            y = area.y + 8 + i * row_h - scroll
            rects.append((i, pygame.Rect(area.x + 8, y, area.width - 16, row_h - 8)))
        return rects

    def _draw_list_row(self, rect: pygame.Rect, label: str, active: bool, hover: bool) -> None:
        assert self._screen is not None
        if active:
            bg = self._colors["button_active"]
        elif hover:
            bg = self._colors["button_hover"]
        else:
            bg = self._colors["button"]
        border = self._colors["accent"] if (active or hover) else self._colors["button_border"]
        pygame.draw.rect(self._screen, bg, rect, border_radius=8)
        pygame.draw.rect(self._screen, border, rect, width=2, border_radius=8)
        font = self._make_font(19, bold=active)
        text = font.render(label, True, self._colors["panel_text"])
        self._screen.blit(text, (rect.x + 14, rect.centery - text.get_height() // 2))

    def _select_models(
        self, mode: str, color: Player | None, dev: bool, review: bool, models: list[ModelOption]
    ) -> MatchSetup | str | None:
        """Second menu screen: pick which model plays each AI seat.

        Returns the finished :class:`MatchSetup`, the sentinel ``"back"`` (return to
        the main menu), or ``None`` if the window was closed. With no models to choose
        from, returns a setup with unset keys (the caller falls back to its default).
        """
        assert self._screen is not None
        assert self._clock is not None
        if not models:
            return MatchSetup(mode, color, dev, review)

        sel = {"white": 0, "black": 0, "ai": 0}
        scroll = {"white": 0, "black": 0, "ai": 0}
        cx = self.window_width // 2
        row_h = 42
        area_top, area_bottom = 150, self.window_height - 130

        title_font = self._make_font(46, bold=True)
        sub_font = self._make_font(21, bold=False)
        col_font = self._make_font(22, bold=True)

        while True:
            mouse = pygame.mouse.get_pos()
            if mode == "vs_ai":
                cols = [("ai", pygame.Rect(cx - 330, area_top, 660, area_bottom - area_top))]
                col_labels = {"ai": "IA adverse"}
            else:
                cw = 430
                cols = [
                    ("white", pygame.Rect(cx - cw - 14, area_top, cw, area_bottom - area_top)),
                    ("black", pygame.Rect(cx + 14, area_top, cw, area_bottom - area_top)),
                ]
                col_labels = {"white": "IA Blancs", "black": "IA Noirs"}

            back_btn = pygame.Rect(cx - 320, self.window_height - 92, 300, 56)
            start_btn = pygame.Rect(cx + 20, self.window_height - 92, 300, 56)

            row_rects = {
                ckey: self._model_row_rects(area, len(models), row_h, scroll[ckey])
                for ckey, area in cols
            }
            max_scroll = {
                ckey: max(0, len(models) * row_h - (area.height - 16)) for ckey, area in cols
            }

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return "back"
                if event.type == pygame.MOUSEWHEEL:
                    for ckey, area in cols:
                        if area.collidepoint(mouse):
                            scroll[ckey] = min(
                                max_scroll[ckey], max(0, scroll[ckey] - event.y * row_h)
                            )
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    if back_btn.collidepoint(pos):
                        return "back"
                    if start_btn.collidepoint(pos):
                        if mode == "vs_ai":
                            return MatchSetup(
                                mode, color, dev, review, ai_model=models[sel["ai"]].key
                            )
                        return MatchSetup(
                            mode,
                            color,
                            dev,
                            review,
                            white_model=models[sel["white"]].key,
                            black_model=models[sel["black"]].key,
                        )
                    for ckey, area in cols:
                        if not area.collidepoint(pos):
                            continue
                        for idx, rect in row_rects[ckey]:
                            if rect.collidepoint(pos):
                                sel[ckey] = idx

            self._draw_gradient_background()
            title = title_font.render("Choix des modeles", True, self._colors["panel_text"])
            self._screen.blit(title, title.get_rect(center=(cx, 64)))
            hint = (
                "Molette pour derouler la liste. Echap: retour."
                if len(models) * row_h > (area_bottom - area_top - 16)
                else "Cliquez un modele pour le selectionner."
            )
            sub = sub_font.render(hint, True, self._colors["panel_subtext"])
            self._screen.blit(sub, sub.get_rect(center=(cx, 108)))

            for ckey, area in cols:
                label = col_font.render(col_labels[ckey], True, self._colors["accent"])
                self._screen.blit(label, (area.x + 4, area.y - 34))
                pygame.draw.rect(self._screen, self._colors["panel"], area, border_radius=12)
                pygame.draw.rect(
                    self._screen, self._colors["button_border"], area, width=1, border_radius=12
                )
                prev_clip = self._screen.get_clip()
                self._screen.set_clip(area)
                for idx, rect in row_rects[ckey]:
                    if rect.bottom < area.y or rect.top > area.bottom:
                        continue
                    self._draw_list_row(
                        rect,
                        models[idx].label,
                        active=idx == sel[ckey],
                        hover=rect.collidepoint(mouse) and area.collidepoint(mouse),
                    )
                self._screen.set_clip(prev_clip)

            self._draw_button(back_btn, "Retour", hover=back_btn.collidepoint(mouse))
            self._draw_button(start_btn, "Commencer", hover=start_btn.collidepoint(mouse))

            pygame.display.flip()
            self._clock.tick(60)

    # ---------------------------------------------------- in-game step / end
    def wait_for_step(
        self,
        view_as: Player | None = None,
        analysis_arrows: list[MoveArrow] | None = None,
        status_lines: list[str] | None = None,
        badge: MoveBadge | None = None,
        label: str = "Coup suivant  >",
    ) -> bool:
        """Block until the user asks for the next move (click the button under the
        move list, or press Space/Enter/Right). Returns ``False`` if the window was
        closed."""
        assert self._screen is not None
        assert self._clock is not None
        if not self.layout.step_button:  # stepping without having said so up front
            self.set_layout(replace(self.layout, step_button=True))
        btn = self._sidebar()["step"]
        step_keys = (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT)
        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                self._scroll_history(event)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                    if not self.show_legend(view_as=view_as):
                        return False
                    continue
                if event.type == pygame.KEYDOWN and event.key in step_keys:
                    return True
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and self._legend_hit(event.pos)
                ):
                    if not self.show_legend(view_as=view_as):
                        return False
                    continue
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and btn.collidepoint(event.pos)
                ):
                    return True
            self._draw_frame(
                view_as=view_as,
                analysis_arrows=analysis_arrows,
                status_lines=status_lines,
                badge=badge,
            )
            self._draw_button(btn, label, hover=btn.collidepoint(mouse), font_size=20)
            pygame.display.flip()
            self._clock.tick(60)

    def show_game_over(self, message: str, view_as: Player | None = None) -> bool:
        """Dim the board and show the result over a "back to menu" button.

        ``message`` may span several lines — the result, then whatever the caller
        wants to add after it (a move-review summary, say) — and the banner grows
        to fit them.

        Returns ``True`` to go back to the menu (button click or Enter/Space/Esc),
        ``False`` if the window was closed.
        """
        assert self._screen is not None
        assert self._clock is not None
        cx, cy = self.window_width // 2, self.window_height // 2
        lines = [line for line in message.splitlines() if line.strip()][:6]
        banner_height = 190 + 32 * len(lines)
        banner = pygame.Rect(cx - 300, cy - banner_height // 2, 600, banner_height)
        btn = pygame.Rect(cx - 150, banner.bottom - 82, 300, 58)
        title_font = self._make_font(44, bold=True)
        msg_font = self._make_font(24, bold=False)
        back_keys = (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE)
        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                self._scroll_history(event)
                if event.type == pygame.KEYDOWN and event.key in back_keys:
                    return True
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and btn.collidepoint(event.pos)
                ):
                    return True

            self._draw_frame(view_as=view_as)
            dim = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
            dim.fill((10, 14, 24, 190))
            self._screen.blit(dim, (0, 0))
            pygame.draw.rect(self._screen, self._colors["panel"], banner, border_radius=18)
            pygame.draw.rect(
                self._screen, self._colors["accent"], banner, width=2, border_radius=18
            )
            title = title_font.render("Partie terminee", True, self._colors["panel_text"])
            self._screen.blit(title, title.get_rect(center=(cx, banner.y + 52)))
            for i, line in enumerate(lines):
                msg = msg_font.render(line, True, self._colors["panel_subtext"])
                self._screen.blit(msg, msg.get_rect(center=(cx, banner.y + 106 + i * 32)))
            self._draw_button(btn, "Retour au menu", hover=btn.collidepoint(mouse), font_size=22)
            pygame.display.flip()
            self._clock.tick(60)

    def _choose_promotion(
        self,
        source: Coord,
        dest: Coord,
        choices: list[PieceType],
        view_as: Player | None,
        fps: int = 60,
    ) -> tuple[bool, PieceType | None]:
        """Ask which piece the pawn on ``source`` becomes on ``dest``.

        Returns ``(closed, piece)``: ``closed`` when the window was shut, and ``piece``
        ``None`` when the choice was dismissed (Esc, right click, or a click off the
        picker), which takes the move back rather than queening behind the player's
        back.
        """
        assert self.game is not None
        assert self._screen is not None
        assert self._clock is not None
        mover = self.game.turn
        cells = promotion_picker_cells(self._to_display_coord(dest, view_as=view_as))
        options = [
            (self._coord_to_rect(cell), piece)
            for cell, piece in zip(cells, PROMOTION_PIECES, strict=True)
            if piece in choices
        ]
        board = pygame.Rect(self.margin, self.margin, self.board_size_px, self.board_size_px)

        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True, None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return False, None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                    return False, None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, piece in options:
                        if rect.collidepoint(event.pos):
                            return False, piece
                    return False, None

            self._draw_frame(view_as=view_as, selected_coord=source)
            dim = pygame.Surface(board.size, pygame.SRCALPHA)
            dim.fill((10, 14, 24, 150))
            self._screen.blit(dim, board.topleft)
            for rect, piece in options:
                hover = rect.collidepoint(mouse)
                card = rect.inflate(-6, -6)
                fill = (255, 255, 255) if hover else (226, 231, 240)
                pygame.draw.rect(self._screen, fill, card, border_radius=12)
                border = self._colors["accent"] if hover else self._colors["board_border"]
                pygame.draw.rect(self._screen, border, card, width=3, border_radius=12)
                sprite = self._piece_sprites[(mover, piece)]
                self._screen.blit(sprite, sprite.get_rect(center=rect.center))
            pygame.display.flip()
            self._clock.tick(fps)

    def request_move(
        self,
        view_as: Player | None = None,
        fps: int = 60,
        analysis_arrows: list[MoveArrow] | None = None,
        status_lines: list[str] | None = None,
        badge: MoveBadge | None = None,
    ) -> Move | None:
        """Internal method: waits for a single move without closing pygame.

        ``analysis_arrows``/``status_lines`` feed the developer overlay so the
        model's read of the position stays visible while the human deliberates.

        A pawn reaching the last rank opens a picker for the piece it becomes.

        Returns:
                The move in engine coordinates, promotion included, or None if the
                window was closed.
        """
        if self.game is None:
            raise ValueError("No ChessGame assigned. Use set_game(...) first.")

        selected_source: tuple[int, int] | None = None
        possible_destinations: set[tuple[int, int]] = set()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                self._scroll_history(event)

                if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                    if not self.show_legend(view_as=view_as):
                        return None
                    continue

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self._legend_hit(event.pos):
                        if not self.show_legend(view_as=view_as):
                            return None
                        continue

                    display_coord = self._pixel_to_display_coord(event.pos)
                    if display_coord is None:
                        continue

                    real_coord = self._to_real_coord(display_coord, view_as=view_as)
                    piece = self.game.grid[real_coord[0]][real_coord[1]]

                    if selected_source is None:
                        if piece is not None and piece.player == self.game.turn:
                            selected_source = real_coord
                            moves = self.game.possible_moves(selected_source)
                            possible_destinations = set(moves)
                        continue

                    if real_coord == selected_source:
                        selected_source = None
                        possible_destinations = set()
                        continue

                    if piece is not None and piece.player == self.game.turn:
                        selected_source = real_coord
                        moves = self.game.possible_moves(selected_source)
                        possible_destinations = set(moves)
                        continue

                    if real_coord in possible_destinations:
                        choices = self.game.promotion_choices(selected_source, real_coord)
                        if not choices:
                            return Move(selected_source, real_coord)
                        closed, promotion = self._choose_promotion(
                            selected_source, real_coord, choices, view_as=view_as, fps=fps
                        )
                        if closed:
                            return None
                        if promotion is not None:
                            return Move(selected_source, real_coord, promotion)
                        selected_source = None  # dismissed: take the move back
                        possible_destinations = set()

            self._draw_frame(
                view_as=view_as,
                selected_coord=selected_source,
                possible_destinations=possible_destinations,
                analysis_arrows=analysis_arrows,
                status_lines=status_lines,
                badge=badge,
            )
            pygame.display.flip()
            self._clock.tick(fps)

    def play_game(self, view_as: Player | None = None, fps: int = 60) -> None:
        """Launch a complete interactive game loop in a persistent window.

        The window stays open throughout the entire game. The game loop continues
        until the window is closed or a king is captured. No window reloads.

        Args:
                view_as: Whose side sits at the bottom; None follows the side to move.
                fps: Max refresh rate.
        """
        if self.game is None:
            raise ValueError("No ChessGame assigned. Use set_game(...) first.")

        self._ensure_initialized()
        assert self._screen is not None
        assert self._clock is not None

        while True:
            move = self.request_move(view_as=view_as, fps=fps)
            if move is None:
                # Window closed
                break

            source, dest = move[0], move[1]
            self.game.play(*move)
            print(f"Coup joue: {source} -> {dest}")

            if self.game.is_checkmate():
                # game.turn is the side that has just been mated.
                winner = self.game.turn.opponent
                print(f"Partie terminee: echec et mat. {winner.name} gagne!")
                break

        pygame.quit()
        self._initialized = False
