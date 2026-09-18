"""Safety: the blunders that decide games.

The rules no longer let a model hang its own king — that move is simply illegal — so
these themes ask the question one rung up: which moves lose the game, or a piece, for
nothing.

- ``escape_check`` — the king is attacked: of the replies the rules allow, play one
  that does not walk straight into mate;
- ``avoid_mate`` — not in check, but many moves allow mate in one: play none of them;
- ``avoid_piece_hang`` — many moves put a piece en prise for nothing.
"""

from __future__ import annotations

import random

from kaisparov.bench.generators.base import (
    HEAVY_TYPES,
    PIECE_TYPES,
    BoardBuilder,
    SamplingGenerator,
    is_quiet,
    near,
)
from kaisparov.bench.oracle import WIN, Oracle
from kaisparov.bench.problem import Problem
from kaisparov.bench.tasks import AvoidMoves, FindMove
from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import PieceType, Player


def _crowded_king(rng: random.Random, attackers: int, defenders: int) -> BoardBuilder | None:
    """White's king with enemy pieces gathered around it — where king safety is tested."""
    board = BoardBuilder(rng)
    king = board.place(Player.WHITE, PieceType.KING)
    if king is None or board.place(Player.BLACK, PieceType.KING) is None:
        return None
    zone = near(king, 3)
    if not board.place_all(
        Player.BLACK, [rng.choice(HEAVY_TYPES) for _ in range(attackers)], squares=zone
    ):
        return None
    if not board.place_all(
        Player.WHITE, [rng.choice(PIECE_TYPES) for _ in range(defenders)], squares=zone
    ):
        return None
    return board


def _share_bucket(fraction: float) -> int:
    """Difficulty from the share of good moves: the fewer, the harder."""
    return 3 if fraction <= 0.1 else 2 if fraction <= 0.25 else 1


class EscapeCheckGenerator(SamplingGenerator):
    name = "escape_check"
    theme = "escape_check"
    description = "The king is attacked: get out of it without walking into mate."

    def __init__(self, max_fraction: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_fraction = max_fraction
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        board = _crowded_king(rng, rng.randint(1, 3), rng.randint(0, 3))
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not game.is_in_check(Player.WHITE) or game.is_in_check(Player.BLACK):
            return None  # White in check; Black in check too would be an illegal position
        if self.oracle.is_over(game):
            return None
        legal = game.legal_moves()
        safe = self.oracle.safe_moves(game)
        if not safe or len(safe) > self.max_fraction * len(legal):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=FindMove.of(safe),
            difficulty=_share_bucket(len(safe) / len(legal)),
            meta={"safe": len(safe), "legal": len(legal)},
        )


class AvoidMateGenerator(SamplingGenerator):
    name = "avoid_mate"
    theme = "avoid_mate"
    description = "Not in check, but many moves allow mate in one: play none of them."

    def __init__(self, min_fraction: float = 0.2, min_forbidden: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.min_fraction, self.min_forbidden = min_fraction, min_forbidden
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        board = _crowded_king(rng, rng.randint(2, 4), rng.randint(1, 4))
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle):
            return None
        legal = game.legal_moves()
        safe = set(self.oracle.safe_moves(game))
        losing = [m for m in legal if m not in safe]
        if not safe or len(losing) < max(self.min_forbidden, self.min_fraction * len(legal)):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=AvoidMoves.of(losing),
            difficulty=_share_bucket(len(safe) / len(legal)),
            meta={"losing": len(losing), "legal": len(legal)},
        )


class AvoidPieceHangGenerator(SamplingGenerator):
    """Material is level and nothing hangs yet; several moves would give a piece away.

    Screening every move at the full depth costs too much, so moves are screened one
    reply deep and only the suspects are confirmed at ``plies``: a forbidden move is
    always a proven loss, and a loss the screen misses only makes the problem easier.
    """

    name = "avoid_piece_hang"
    theme = "avoid_piece_hang"
    description = "Several moves put a piece en prise for nothing: play none of them."

    def __init__(self, min_loss: float = 3.0, plies: int = 3, min_forbidden: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.min_loss, self.plies, self.min_forbidden = min_loss, plies, min_forbidden
        self.oracle = Oracle()

    def propose(self, rng: random.Random) -> Problem | None:
        white = [rng.choice(PIECE_TYPES) for _ in range(rng.randint(3, 6))]
        black = [rng.choice(PIECE_TYPES) for _ in range(rng.randint(3, 6))]
        board = BoardBuilder.random_board(rng, white, black)
        if board is None:
            return None
        position = board.position(Player.WHITE)
        game = position.to_game()
        if not is_quiet(game, self.oracle):
            return None
        screen = self.oracle.material_gains(game, 1)
        if max(screen.values()) > 0.5 or max(screen.values()) < -0.5:
            return None  # something already hangs, for one side or the other
        forbidden = [
            move
            for move, gain in screen.items()
            if -WIN < gain <= -self.min_loss and -WIN < self._confirm(game, move) <= -self.min_loss
        ]
        if len(forbidden) < self.min_forbidden or len(forbidden) >= len(screen):
            return None
        return Problem(
            id="",
            theme=self.theme,
            position=position,
            task=AvoidMoves.of(forbidden),
            difficulty=2 if len(forbidden) >= len(screen) / 4 else 1,  # more traps, harder
            meta={"forbidden": len(forbidden), "legal": len(screen)},
        )

    def _confirm(self, game: ChessGame, move) -> float:
        return self.oracle.material_gain(game, move, self.plies)


__all__ = ["AvoidMateGenerator", "AvoidPieceHangGenerator", "EscapeCheckGenerator"]
