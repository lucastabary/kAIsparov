"""A small Gym-like environment wrapping :class:`ChessGame`.

This is the single place that knows how a chess position turns into a
reinforcement-learning transition: which moves are legal, what a move is worth,
and when the game is over. Evaluation (the arena) and future training share this
one definition instead of re-deriving reward/terminal logic each time.

The observation is the live :class:`ChessGame`; neural agents graphify it while
baseline agents read the grid directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from kaisparov.agents.base import Move
from kaisparov.core.board import ChessGame, Undo
from kaisparov.core.draw import DEFAULT_RULES, DrawRules, draw_reason
from kaisparov.core.movegen import all_moves
from kaisparov.core.pieces import Piece, PieceType, Player
from kaisparov.core.utils import get_piece_value

Grid = list[list["Piece | None"]]


@dataclass
class StepResult:
    obs: ChessGame
    reward: float
    done: bool
    info: dict


class ChessEnv:
    """Two-player, capture-the-king chess environment.

    Both sides' moves go through :meth:`step`; ``reward`` is always from the point
    of view of the player who just moved.

    A game ends on a king capture, when the side to move has no move, on a draw
    rule (repetition / no progress / insufficient material, see
    :mod:`kaisparov.core.draw`) or at ``max_plies``. ``end_reason`` names which.
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

    def reset(self, board: Grid | None = None, turn: Player = Player.WHITE) -> ChessGame:
        self.game = ChessGame(initial_board=board, turn=turn)
        self.done = False
        self.winner = None
        self.end_reason = None
        self.plies = 0
        return self.game

    def legal_moves(self) -> list[Move]:
        return all_moves(self.game.grid, self.game.turn, self.game.en_passant_target)

    def step(self, move: Move) -> StepResult:
        if self.done:
            raise RuntimeError("step() called on a finished game; call reset() first.")

        source, dest = move
        if not self.game.is_move_valid(source, dest):
            raise ValueError(f"illegal move {source} -> {dest}")

        mover = self.game.turn
        undo: Undo = self.game.make(source, dest)
        captured = undo.captured

        # Reward is material only: a draw — however it is reached — scores 0, exactly
        # like any other non-capturing move. Ending a game must never be a reward
        # event in itself, or the agent would learn to chase (or flee) draws.
        reward = get_piece_value(captured.type) if captured is not None else 0.0
        king_captured = captured is not None and captured.type == PieceType.KING

        self.plies += 1
        info: dict = {"captured": captured, "mover": mover}

        if king_captured:
            self.done = True
            self.winner = mover
            self.end_reason = "king_captured"
        elif not self.legal_moves():
            self.done = True  # opponent has no move: stalemate-like draw
            self.winner = None
            self.end_reason = "stalemate"
        elif (drawn := draw_reason(self.game, self.draw_rules)) is not None:
            self.done = True
            self.winner = None
            self.end_reason = drawn
        elif self.plies >= self.max_plies:
            self.done = True
            self.winner = None
            self.end_reason = "max_plies"

        info["winner"] = self.winner
        info["end_reason"] = self.end_reason
        return StepResult(obs=self.game, reward=reward, done=self.done, info=info)
