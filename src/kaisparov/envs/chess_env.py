"""A small Gym-like environment wrapping :class:`ChessGame`.

This is the single place that knows how a chess position turns into a
reinforcement-learning transition: which moves are legal, what a move is worth,
and when the game is over. Evaluation (the arena) and training share this one
definition instead of re-deriving reward/terminal logic each time.

The observation is the live :class:`ChessGame`; neural agents graphify it while
baseline agents read the grid directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaisparov.core.draw import DEFAULT_RULES, DrawRules, draw_reason
from kaisparov.core.game import ChessGame, Undo
from kaisparov.core.move import Move
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.core.utils import get_piece_value

Grid = list[list["Piece | None"]]

CHECKMATE = "checkmate"
MAX_PLIES = "max_plies"


@dataclass
class StepResult:
    obs: ChessGame
    reward: float
    done: bool
    info: dict


class ChessEnv:
    """Two-player standard-chess environment.

    Both sides' moves go through :meth:`step`; ``reward`` is always from the point
    of view of the player who just moved.

    A game ends on checkmate, on a draw rule (stalemate / repetition / no progress /
    insufficient material, see :mod:`kaisparov.core.draw`) or at ``max_plies``.
    ``end_reason`` names which.
    """

    def __init__(self, max_plies: int = 200, draw_rules: DrawRules | None = DEFAULT_RULES):
        self.max_plies = max_plies
        # Which draw rules end a game (``None`` = none; see kaisparov.core.draw).
        self.draw_rules = draw_rules
        self.game = ChessGame()
        self.done = False
        self.winner: Player | None = None
        self.end_reason: str | None = None
        self.plies = 0

    def reset(
        self,
        board: Grid | None = None,
        turn: Player = Player.WHITE,
        *,
        game: ChessGame | None = None,
    ) -> ChessGame:
        """Start a new episode from ``board`` — or from ``game`` as it stands, history
        included (en-passant target, repetition record), which a set-up benchmark
        position needs. The env plays on ``game`` itself rather than a copy."""
        self.game = game if game is not None else ChessGame(initial_board=board, turn=turn)
        self.done = False
        self.winner = None
        self.end_reason = None
        self.plies = 0
        return self.game

    def legal_moves(self) -> list[Move]:
        return self.game.legal_moves()

    def step(self, move: Move) -> StepResult:
        if self.done:
            raise RuntimeError("step() called on a finished game; call reset() first.")

        source, dest = move[0], move[1]
        promotion = move[2] if len(move) > 2 else None
        if not self.game.is_move_valid(source, dest, promotion):
            raise ValueError(f"illegal move {source} -> {dest}")

        mover = self.game.turn
        undo: Undo = self.game.make(source, dest, promotion)
        captured = undo.captured

        # Reward is material only: a draw — however it is reached — scores 0, exactly
        # like any other quiet move. Ending a game must never be a reward event in
        # itself, or the agent would learn to chase (or flee) draws. Checkmate is the
        # exception the trainer adds on top (see kaisparov.training.reward); here the
        # env stays a pure material ledger.
        reward = get_piece_value(captured.type) if captured is not None else 0.0
        if undo.move.promotion is not None:
            # A promotion is a material event with no capture: the pawn is gone and
            # something far better stands in its place.
            reward += get_piece_value(undo.move.promotion) - get_piece_value(PieceType.PAWN)

        self.plies += 1
        info: dict = {"captured": captured, "mover": mover, "promotion": undo.move.promotion}

        if self.game.is_checkmate():
            self.done = True
            self.winner = mover
            self.end_reason = CHECKMATE
        elif (drawn := draw_reason(self.game, self.draw_rules)) is not None:
            self.done = True
            self.winner = None
            self.end_reason = drawn
        elif not self.game.legal_moves():
            # Nothing left to play and it is not mate: stalemate, even when the rule
            # that names it has been switched off.
            self.done = True
            self.winner = None
            self.end_reason = "stalemate"
        elif self.plies >= self.max_plies:
            self.done = True
            self.winner = None
            self.end_reason = MAX_PLIES

        info["winner"] = self.winner
        info["end_reason"] = self.end_reason
        return StepResult(obs=self.game, reward=reward, done=self.done, info=info)
