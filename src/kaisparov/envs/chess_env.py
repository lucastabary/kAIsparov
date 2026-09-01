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
    """

    def __init__(self, max_plies: int = 200):
        self.max_plies = max_plies
        self.game = ChessGame()
        self.done = False
        self.winner: Player | None = None
        self.plies = 0

    def reset(self, board: Grid | None = None, turn: Player = Player.WHITE) -> ChessGame:
        self.game = ChessGame(initial_board=board, turn=turn)
        self.done = False
        self.winner = None
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

        reward = get_piece_value(captured.type) if captured is not None else 0.0
        king_captured = captured is not None and captured.type == PieceType.KING

        self.plies += 1
        info: dict = {"captured": captured, "mover": mover}

        if king_captured:
            self.done = True
            self.winner = mover
        elif not self.legal_moves():
            self.done = True  # opponent has no move: stalemate-like draw
            self.winner = None
        elif self.plies >= self.max_plies:
            self.done = True
            self.winner = None

        info["winner"] = self.winner
        return StepResult(obs=self.game, reward=reward, done=self.done, info=info)
