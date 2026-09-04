"""Negamax alpha-beta search that uses the model's critic to evaluate leaves and its
actor to order moves (better ordering -> more pruning -> deeper search per cost).

This is the search-improved player: it looks ahead ``depth`` plies and picks the move
leading to the best critic-evaluated position, correcting the raw policy's local
errors. It's the natural bridge toward an AlphaZero-style setup.

Cost note: it runs one model forward per visited node, so it's meant for evaluation,
analysis, and playing a strong opponent — not for bulk self-play data collection on
CPU. Keep ``depth`` small (2-3). ``depth=1`` is a cheap 1-ply critic lookahead.
"""

from __future__ import annotations

import torch

from kaisparov.agents.base import Move
from kaisparov.core.board import ChessGame
from kaisparov.core.movegen import all_moves
from kaisparov.core.pieces import PieceType
from kaisparov.core.utils import coord_to_index

WIN = 1e6  # value of capturing the king (dominates any critic value)


class MinimaxAgent:
    name = "minimax"

    def __init__(self, model: torch.nn.Module, processor, depth: int = 2):
        self.model = model
        self.processor = processor
        self.depth = depth
        # Map a move (src, dst) to its position in the actor's edge scores, for ordering.
        edge_index = processor.static_graph_edges[0]
        self._edge_pos = {
            (int(edge_index[0, k]), int(edge_index[1, k])): k for k in range(edge_index.shape[1])
        }

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _forward(self, game: ChessGame):
        """One forward pass -> (action_scores over edges, state value for side to move)."""
        data = self.processor.graphify(game).to(self._device())
        action_scores, value = self.model(data)
        return action_scores, float(value.reshape(-1)[0])

    def _order(self, moves: list[Move], action_scores: torch.Tensor) -> list[Move]:
        def score(mv: Move) -> float:
            pos = self._edge_pos.get((coord_to_index(mv[0]), coord_to_index(mv[1])))
            return float(action_scores[pos]) if pos is not None else -1e9

        return sorted(moves, key=score, reverse=True)

    def _search(self, game: ChessGame, depth: int, alpha: float, beta: float) -> float:
        action_scores, value = self._forward(game)
        if depth == 0:
            return value
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        if not moves:
            return value  # no move available -> evaluate statically

        best = -WIN * 2
        for mv in self._order(moves, action_scores):
            undo = game.make(*mv)
            if undo.captured is not None and undo.captured.type == PieceType.KING:
                child = WIN  # this move wins outright
            else:
                child = -self._search(game, depth - 1, -beta, -alpha)
            game.unmake(undo)

            if child > best:
                best = child
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # opponent won't allow this line
        return best

    def select_move(self, game: ChessGame) -> Move | None:
        moves = all_moves(game.grid, game.turn, game.en_passant_target)
        if not moves:
            return None

        self.model.eval()
        with torch.no_grad():
            action_scores, _ = self._forward(game)
            best_move, best_val, alpha = None, -WIN * 2, -WIN * 2
            for mv in self._order(moves, action_scores):
                undo = game.make(*mv)
                if undo.captured is not None and undo.captured.type == PieceType.KING:
                    value = WIN
                else:
                    value = -self._search(game, self.depth - 1, -WIN * 2, -alpha)
                game.unmake(undo)

                if value > best_val:
                    best_val, best_move = value, mv
                if value > alpha:
                    alpha = value
        return best_move
