from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from kaisparov.core.board import ChessGame
from kaisparov.core.coords import Coord
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
    human_color: Player = Player.WHITE  # only meaningful for "vs_ai"
    dev_mode: bool = False  # surface the model's analysis while playing
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


class GameInterface:
    """Graphical interface for rendering a ChessGame in a separate window."""

    def __init__(self, game: ChessGame | None = None):
        self.game: ChessGame | None = game

        self.cell_size = 88
        self.board_size_px = self.cell_size * BOARD_SIZE
        self.margin = 34
        self.panel_width = 250
        self.window_width = self.board_size_px + self.margin * 2 + self.panel_width
        self.window_height = self.board_size_px + self.margin * 2

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

    def set_game(self, game: ChessGame) -> None:
        """Assigns a ChessGame instance to this interface."""
        self.game = game

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

    def _last_move_display_cells(self, use_pov: bool) -> set[tuple[int, int]]:
        """Display coords of the last move's from/to squares (empty if no move yet)."""
        if self.game is None or self.game.last_move is None:
            return set()
        return {self._to_display_coord(square, use_pov=use_pov) for square in self.game.last_move}

    def _draw_gradient_background(self) -> None:
        assert self._screen is not None
        for y in range(self.window_height):
            t = y / max(1, self.window_height - 1)
            r = int(self._colors["bg_start"][0] * (1 - t) + self._colors["bg_end"][0] * t)
            g = int(self._colors["bg_start"][1] * (1 - t) + self._colors["bg_end"][1] * t)
            b = int(self._colors["bg_start"][2] * (1 - t) + self._colors["bg_end"][2] * t)
            pygame.draw.line(self._screen, (r, g, b), (0, y), (self.window_width, y))

    def _draw_board(self, use_pov: bool) -> None:
        assert self._screen is not None
        assert self.game is not None

        board_left = self.margin
        board_top = self.margin

        grid = self.game.get_pov_grid() if use_pov else self.game.grid

        board_rect = pygame.Rect(
            board_left - 4, board_top - 4, self.board_size_px + 8, self.board_size_px + 8
        )
        pygame.draw.rect(self._screen, self._colors["board_border"], board_rect, border_radius=12)

        last_move_cells = self._last_move_display_cells(use_pov)

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
        use_pov: bool,
    ) -> None:
        assert self._screen is not None

        if selected_coord is not None:
            x, y = self._to_display_coord(selected_coord, use_pov=use_pov)
            rect = self._coord_to_rect((x, y))
            pygame.draw.rect(self._screen, (245, 214, 71), rect, width=5, border_radius=9)

        for dest in possible_destinations:
            x, y = self._to_display_coord(dest, use_pov=use_pov)
            rect = self._coord_to_rect((x, y))
            pygame.draw.circle(self._screen, (82, 196, 26), rect.center, 11)
            pygame.draw.circle(self._screen, (242, 255, 233), rect.center, 11, width=2)

    def _draw_analysis_overlay(self, arrows: list[MoveArrow], use_pov: bool) -> None:
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
            src_disp = self._to_display_coord(arrow.source, use_pov=use_pov)
            dst_disp = self._to_display_coord(arrow.dest, use_pov=use_pov)
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

    def _to_real_coord(self, display_coord: tuple[int, int], use_pov: bool) -> tuple[int, int]:
        assert self.game is not None
        if use_pov:
            return self.game.from_pov_coord(display_coord)
        return display_coord

    def _to_display_coord(self, real_coord: tuple[int, int], use_pov: bool) -> tuple[int, int]:
        assert self.game is not None
        if use_pov:
            return self.game.to_pov_coord(real_coord)
        return real_coord

    def _draw_side_panel(self, use_pov: bool, status_lines: list[str] | None = None) -> None:
        assert self._screen is not None
        assert self.game is not None

        panel_left = self.margin + self.board_size_px + 16
        panel_rect = pygame.Rect(panel_left, self.margin, self.panel_width - 16, self.board_size_px)
        pygame.draw.rect(self._screen, self._colors["panel"], panel_rect, border_radius=16)
        pygame.draw.rect(
            self._screen, self._colors["accent"], panel_rect, width=2, border_radius=16
        )

        title_font = self._make_font(34, bold=True)
        label_font = self._make_font(22, bold=True)
        value_font = self._make_font(30, bold=True)
        small_font = self._make_font(18, bold=False)

        title = title_font.render("kAIsparov", True, self._colors["panel_text"])
        self._screen.blit(title, (panel_left + 18, self.margin + 18))

        turn_label = label_font.render("Tour", True, self._colors["panel_subtext"])
        turn_value = value_font.render(self.game.turn.name, True, self._colors["panel_text"])

        count_label = label_font.render("Coup", True, self._colors["panel_subtext"])
        count_value = value_font.render(str(self.game.count), True, self._colors["panel_text"])

        self._screen.blit(turn_label, (panel_left + 20, self.margin + 90))
        self._screen.blit(turn_value, (panel_left + 20, self.margin + 118))
        self._screen.blit(count_label, (panel_left + 20, self.margin + 180))
        self._screen.blit(count_value, (panel_left + 20, self.margin + 208))

        if status_lines:
            status_font = self._make_font(19, bold=False)
            y = self.margin + 270
            for line in status_lines[:6]:
                text = status_font.render(line, True, self._colors["panel_subtext"])
                self._screen.blit(text, (panel_left + 20, y))
                y += 28

        view_mode = "POV courant" if use_pov else "Vue absolue"
        view_text = small_font.render(
            f"Affichage: {view_mode}", True, self._colors["panel_subtext"]
        )
        hint_text = small_font.render(
            "Fermez la fenetre pour quitter.", True, self._colors["panel_subtext"]
        )
        self._screen.blit(view_text, (panel_left + 20, self.margin + self.board_size_px - 72))
        self._screen.blit(hint_text, (panel_left + 20, self.margin + self.board_size_px - 44))

    def _draw_frame(
        self,
        use_pov: bool,
        selected_coord: tuple[int, int] | None = None,
        possible_destinations: set[tuple[int, int]] | None = None,
        analysis_arrows: list[MoveArrow] | None = None,
        status_lines: list[str] | None = None,
    ) -> None:
        self._draw_gradient_background()
        self._draw_board(use_pov=use_pov)
        self._draw_analysis_overlay(analysis_arrows or [], use_pov=use_pov)
        self._draw_selection_overlay(
            selected_coord=selected_coord,
            possible_destinations=possible_destinations or set(),
            use_pov=use_pov,
        )
        self._draw_side_panel(use_pov=use_pov, status_lines=status_lines)

    def render(self, use_pov: bool = True, fps: int = 60) -> None:
        """Opens a dedicated window and renders the game continuously.

        Args:
                use_pov: If True, render from current player's point of view.
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

            self._draw_frame(use_pov=use_pov)
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
        half = (width - 16) // 2
        return {
            "solo": pygame.Rect(left, 232, width, button_h),
            "vs_ai": pygame.Rect(left, 312, width, button_h),
            "ai_vs_ai": pygame.Rect(left, 392, width, button_h),
            "white": pygame.Rect(left, 512, half, 48),
            "black": pygame.Rect(left + half + 16, 512, half, 48),
            "dev": pygame.Rect(left, 592, width, 44),
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
        and the developer-mode switch persist until a mode is clicked. Solo starts
        immediately; the AI modes open a model-selection screen first so the user
        chooses which trained model plays each AI seat.
        """
        self._ensure_initialized()
        assert self._screen is not None
        assert self._clock is not None

        models = models or []
        color = Player.WHITE
        dev = False
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
                        return MatchSetup("solo", color, dev)
                    if layout["vs_ai"].collidepoint(pos):
                        setup = self._select_models("vs_ai", color, dev, models)
                        if setup != "back":
                            return setup
                    if layout["ai_vs_ai"].collidepoint(pos):
                        setup = self._select_models("ai_vs_ai", color, dev, models)
                        if setup != "back":
                            return setup
                    if layout["white"].collidepoint(pos):
                        color = Player.WHITE
                    if layout["black"].collidepoint(pos):
                        color = Player.BLACK
                    if layout["dev"].collidepoint(pos):
                        dev = not dev

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
            self._draw_button(
                layout["white"],
                "Blancs",
                hover=layout["white"].collidepoint(mouse),
                active=color == Player.WHITE,
            )
            self._draw_button(
                layout["black"],
                "Noirs",
                hover=layout["black"].collidepoint(mouse),
                active=color == Player.BLACK,
            )

            self._draw_checkbox(
                layout["dev"],
                "Mode developpeur (analyse du modele)",
                dev,
                hover=layout["dev"].collidepoint(mouse),
            )

            hint = hint_font.render(
                "Cliquez un mode pour commencer.",
                True,
                self._colors["panel_subtext"],
            )
            self._screen.blit(hint, hint.get_rect(center=(cx, 672)))

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
        self, mode: str, color: Player, dev: bool, models: list[ModelOption]
    ) -> MatchSetup | str | None:
        """Second menu screen: pick which model plays each AI seat.

        Returns the finished :class:`MatchSetup`, the sentinel ``"back"`` (return to
        the main menu), or ``None`` if the window was closed. With no models to choose
        from, returns a setup with unset keys (the caller falls back to its default).
        """
        assert self._screen is not None
        assert self._clock is not None
        if not models:
            return MatchSetup(mode, color, dev)

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
                            return MatchSetup(mode, color, dev, ai_model=models[sel["ai"]].key)
                        return MatchSetup(
                            mode,
                            color,
                            dev,
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
    def _panel_button_rect(self) -> pygame.Rect:
        panel_left = self.margin + self.board_size_px + 16
        return pygame.Rect(panel_left + 20, self.margin + 470, self.panel_width - 56, 54)

    def wait_for_step(
        self,
        use_pov: bool = True,
        analysis_arrows: list[MoveArrow] | None = None,
        status_lines: list[str] | None = None,
        label: str = "Coup suivant  >",
    ) -> bool:
        """Block until the user asks for the next move (click the panel button, or
        press Space/Enter/Right). Returns ``False`` if the window was closed."""
        assert self._screen is not None
        assert self._clock is not None
        btn = self._panel_button_rect()
        step_keys = (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT)
        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key in step_keys:
                    return True
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and btn.collidepoint(event.pos)
                ):
                    return True
            self._draw_frame(
                use_pov=use_pov, analysis_arrows=analysis_arrows, status_lines=status_lines
            )
            self._draw_button(btn, label, hover=btn.collidepoint(mouse), font_size=20)
            pygame.display.flip()
            self._clock.tick(60)

    def show_game_over(self, message: str, use_pov: bool = True) -> bool:
        """Dim the board and show the result over a "back to menu" button.

        Returns ``True`` to go back to the menu (button click or Enter/Space/Esc),
        ``False`` if the window was closed.
        """
        assert self._screen is not None
        assert self._clock is not None
        cx, cy = self.window_width // 2, self.window_height // 2
        banner = pygame.Rect(cx - 270, cy - 130, 540, 260)
        btn = pygame.Rect(cx - 150, cy + 34, 300, 58)
        title_font = self._make_font(44, bold=True)
        msg_font = self._make_font(24, bold=False)
        back_keys = (pygame.K_RETURN, pygame.K_SPACE, pygame.K_ESCAPE)
        while True:
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key in back_keys:
                    return True
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and btn.collidepoint(event.pos)
                ):
                    return True

            self._draw_frame(use_pov=use_pov)
            dim = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
            dim.fill((10, 14, 24, 190))
            self._screen.blit(dim, (0, 0))
            pygame.draw.rect(self._screen, self._colors["panel"], banner, border_radius=18)
            pygame.draw.rect(
                self._screen, self._colors["accent"], banner, width=2, border_radius=18
            )
            title = title_font.render("Partie terminee", True, self._colors["panel_text"])
            self._screen.blit(title, title.get_rect(center=(cx, banner.y + 60)))
            for i, line in enumerate(message.split("\n")[:2]):
                msg = msg_font.render(line, True, self._colors["panel_subtext"])
                self._screen.blit(msg, msg.get_rect(center=(cx, banner.y + 120 + i * 32)))
            self._draw_button(btn, "Retour au menu", hover=btn.collidepoint(mouse), font_size=22)
            pygame.display.flip()
            self._clock.tick(60)

    def _get_single_move(
        self,
        use_pov: bool = True,
        fps: int = 60,
        analysis_arrows: list[MoveArrow] | None = None,
        status_lines: list[str] | None = None,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Internal method: waits for a single move without closing pygame.

        ``analysis_arrows``/``status_lines`` feed the developer overlay so the
        model's read of the position stays visible while the human deliberates.

        Returns:
                (source, destination) in engine coordinates, or None if window closed.
        """
        if self.game is None:
            raise ValueError("No ChessGame assigned. Use set_game(...) first.")

        selected_source: tuple[int, int] | None = None
        possible_destinations: set[tuple[int, int]] = set()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    display_coord = self._pixel_to_display_coord(event.pos)
                    if display_coord is None:
                        continue

                    real_coord = self._to_real_coord(display_coord, use_pov=use_pov)
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
                        return (selected_source, real_coord)

            self._draw_frame(
                use_pov=use_pov,
                selected_coord=selected_source,
                possible_destinations=possible_destinations,
                analysis_arrows=analysis_arrows,
                status_lines=status_lines,
            )
            pygame.display.flip()
            self._clock.tick(fps)

    def request_move(
        self, use_pov: bool = True, fps: int = 60
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Lets the user pick a source and destination square with mouse clicks.

        Returns:
                (source, destination) in engine coordinates, or None if the window is closed.
        """
        if self.game is None:
            raise ValueError("No ChessGame assigned. Use set_game(...) first.")

        self._ensure_initialized()
        assert self._screen is not None
        assert self._clock is not None

        move = self._get_single_move(use_pov=use_pov, fps=fps)

        if move is None:
            pygame.quit()
            self._initialized = False

        return move

    def play_game(self, use_pov: bool = True, fps: int = 60) -> None:
        """Launch a complete interactive game loop in a persistent window.

        The window stays open throughout the entire game. The game loop continues
        until the window is closed or a king is captured. No window reloads.

        Args:
                use_pov: If True, render from current player's point of view.
                fps: Max refresh rate.
        """
        if self.game is None:
            raise ValueError("No ChessGame assigned. Use set_game(...) first.")

        self._ensure_initialized()
        assert self._screen is not None
        assert self._clock is not None

        while True:
            move = self._get_single_move(use_pov=use_pov, fps=fps)
            if move is None:
                # Window closed
                break

            source, dest = move
            captured = self.game.play(source, dest)
            print(f"Coup joue: {source} -> {dest}")

            if captured is not None and captured.type.name == "KING":
                print(f"Partie terminee: roi capture. {self.game.turn.name} gagne!")
                break

        pygame.quit()
        self._initialized = False
