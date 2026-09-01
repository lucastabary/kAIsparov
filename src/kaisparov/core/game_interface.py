from __future__ import annotations

import pygame

from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player

Color = tuple[int, int, int]


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
        self._initialized = True

    def _make_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        candidates = ["segoeuiemoji", "segoe ui symbol", "arial unicode ms", "dejavusans"]
        for name in candidates:
            font = pygame.font.SysFont(name, size, bold=bold)
            if font is not None:
                return font
        return pygame.font.SysFont(None, size, bold=bold)

    def _build_piece_sprites(self) -> None:
        glyph_font = self._make_font(44, bold=True)
        shadow_font = self._make_font(44, bold=True)
        piece_labels = {
            PieceType.KING: "K",
            PieceType.QUEEN: "Q",
            PieceType.BISHOP: "B",
            PieceType.ROOK: "R",
            PieceType.KNIGHT: "N",
            PieceType.PAWN: "P",
        }

        for player in (Player.WHITE, Player.BLACK):
            for piece_type in (
                PieceType.KING,
                PieceType.QUEEN,
                PieceType.BISHOP,
                PieceType.ROOK,
                PieceType.KNIGHT,
                PieceType.PAWN,
            ):
                glyph = piece_labels[piece_type]

                surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                center = (self.cell_size // 2, self.cell_size // 2)

                # Soft token behind the glyph for a sprite-like look and better readability.
                if player == Player.WHITE:
                    token_color = (242, 244, 248, 232)
                    ring_color = (182, 190, 206, 228)
                    glyph_color = (26, 34, 47)
                else:
                    token_color = (39, 52, 72, 238)
                    ring_color = (122, 148, 186, 220)
                    glyph_color = (246, 251, 255)

                pygame.draw.circle(surface, token_color, center, 30)
                pygame.draw.circle(surface, ring_color, center, 30, width=2)

                shadow = shadow_font.render(glyph, True, (0, 0, 0, 170))
                glyph_img = glyph_font.render(glyph, True, glyph_color)

                shadow_rect = shadow.get_rect(center=(center[0] + 2, center[1] + 3))
                glyph_rect = glyph_img.get_rect(center=center)
                surface.blit(shadow, shadow_rect)
                surface.blit(glyph_img, glyph_rect)

                self._piece_sprites[(player, piece_type)] = surface

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

    def _draw_side_panel(self, use_pov: bool) -> None:
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
    ) -> None:
        self._draw_gradient_background()
        self._draw_board(use_pov=use_pov)
        self._draw_selection_overlay(
            selected_coord=selected_coord,
            possible_destinations=possible_destinations or set(),
            use_pov=use_pov,
        )
        self._draw_side_panel(use_pov=use_pov)

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

    def _get_single_move(
        self, use_pov: bool = True, fps: int = 60
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """Internal method: waits for a single move without closing pygame.

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
