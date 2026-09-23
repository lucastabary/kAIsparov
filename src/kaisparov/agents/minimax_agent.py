"""Negamax alpha-beta search over any position evaluator.

``MinimaxAgent`` looks ahead ``depth`` plies and plays the move leading to the best
position, as scored at the leaves by an :class:`~kaisparov.analysis.evaluators.Evaluator`:
material, the heuristic, a trained network's critic
(:class:`~kaisparov.analysis.critic.CriticEvaluator`), or anything later that implements
``evaluate(game)``. The search itself does not care which.

Pruning is only as good as the move order, so ``order`` sorts the moves before they
are searched: captures first by default (biggest gain first), the actor's policy for
a network (:meth:`MinimaxAgent.on_model`, the search-improved player — the bridge to
an AlphaZero-style setup).

As every search here must (see CLAUDE.md), a node with no legal move is scored as
mate (``-WIN``, later mates less bad) or stalemate (0), and so is any drawn position:
a side up a queen does not "win" a stalemate. An evaluator that can say how a move
changed its score without reading the board (``move_delta``, e.g. material) is
tracked move by move from the root instead of re-read at every leaf — what keeps a
material search cheap enough to be a training opponent.

Torch is only imported for a network (:meth:`MinimaxAgent.on_model`): a search on
material or the heuristic stays torch-free.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from kaisparov.agents.base import Move
from kaisparov.agents.safety import safe_moves
from kaisparov.core.draw import DEFAULT_RULES, DrawRules
from kaisparov.core.game import ChessGame
from kaisparov.core.material import WIN, gain_if_played

if TYPE_CHECKING:
    from kaisparov.analysis.evaluators import Evaluator

# Sorts the legal moves of a position, most promising first.
MoveOrder = Callable[[ChessGame, list[Move]], list[Move]]

# Tolerance under which two root scores are a tie (and share the random tie-break).
_TIE = 1e-9


def captures_first(game: ChessGame, moves: list[Move]) -> list[Move]:
    """Captures and promotions first, biggest gain first: they are what prunes."""
    return sorted(moves, key=lambda move: gain_if_played(game, move), reverse=True)


class PolicyOrder:
    """Orders moves by a network's actor scores (one forward pass per call)."""

    def __init__(self, model: Any, processor: Any):
        from kaisparov.core.utils import coord_to_index

        self.model = model
        self.processor = processor
        self._index = coord_to_index
        # Map a move (src, dst) to its position in the actor's edge scores.
        edge_index = processor.static_graph_edges[0]
        self._edge_pos = {
            (int(edge_index[0, k]), int(edge_index[1, k])): k for k in range(edge_index.shape[1])
        }

    def __call__(self, game: ChessGame, moves: list[Move]) -> list[Move]:
        import torch

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        with torch.no_grad():
            scores, _ = self.model(self.processor.graphify(game).to(device))

        def score(move: Move) -> float:
            pos = self._edge_pos.get((self._index(move[0]), self._index(move[1])))
            return float(scores[pos]) if pos is not None else -1e9

        return sorted(moves, key=score, reverse=True)


class MinimaxAgent:
    name = "minimax"

    def __init__(
        self,
        evaluator: Evaluator,
        depth: int = 2,
        *,
        order: MoveOrder | None = None,
        seed: int | None = None,
        avoid_king_suicide: bool = False,
        draw_rules: DrawRules | None = DEFAULT_RULES,
    ):
        if depth < 1:
            raise ValueError(f"MinimaxAgent needs depth >= 1, got {depth}")
        self.evaluator = evaluator
        self.depth = depth
        self.order: MoveOrder = order or captures_first
        self._rng = random.Random(seed)
        # At depth 1 the search does not see the opponent's mating reply; this guard
        # drops root moves that walk into mate in one. Depth >= 2 sees it on its own.
        self.avoid_king_suicide = avoid_king_suicide
        self.draw_rules = draw_rules
        # Incremental scoring, when the evaluator offers it (see the module docstring).
        self._delta: Callable[[Any], float] | None = getattr(evaluator, "move_delta", None)

    @classmethod
    def on_model(cls, model: Any, processor: Any, depth: int = 2, **kwargs: Any) -> MinimaxAgent:
        """The search on a network: its critic scores the leaves, its actor orders moves."""
        from kaisparov.analysis.critic import CriticEvaluator

        return cls(
            CriticEvaluator(model, processor),
            depth,
            order=PolicyOrder(model, processor),
            **kwargs,
        )

    def _search(
        self,
        game: ChessGame,
        depth: int,
        alpha: float,
        beta: float,
        score: float = 0.0,
        ply: int = 0,
    ) -> float:
        """Negamax value for the side to move.

        ``score`` is that side's incremental score when the evaluator has a
        ``move_delta``; otherwise unused, and each leaf is evaluated from the board.
        """
        if game.is_checkmate():
            return -(WIN - ply)  # mated: the later, the less bad
        if game.is_draw(self.draw_rules):
            return 0.0  # stalemate, repetition, dead position: a draw, whatever the count
        if depth == 0:
            return score if self._delta is not None else self.evaluator.evaluate(game)
        best = -2 * WIN
        for move in self.order(game, game.legal_moves()):
            value = self._child(game, move, depth - 1, -beta, -alpha, score, ply + 1)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # the opponent will not allow this line
        return best

    def _child(
        self,
        game: ChessGame,
        move: Move,
        depth: int,
        alpha: float,
        beta: float,
        score: float,
        ply: int,
    ) -> float:
        """Value of ``move`` for the side playing it (negamax of the child position)."""
        undo = game.make(*move)
        child_score = -(score + self._delta(undo)) if self._delta is not None else 0.0
        value = -self._search(game, depth, alpha, beta, child_score, ply)
        game.unmake(undo)
        return value

    def select_move(self, game: ChessGame) -> Move | None:
        moves = game.legal_moves()
        if not moves:
            return None
        if self.avoid_king_suicide:
            moves = safe_moves(game, moves)

        score = self.evaluator.evaluate(game) if self._delta is not None else 0.0
        best_value = -2 * WIN
        best: list[Move] = []
        for move in self.order(game, moves):
            # The window stays just below the best value found, so every move that
            # ties it gets an exact score and a fair share of the random tie-break.
            value = self._child(
                game, move, self.depth - 1, -2 * WIN, -(best_value - _TIE), score, 1
            )
            if value > best_value + _TIE:
                best_value, best = value, [move]
            elif value >= best_value - _TIE:
                best.append(move)
        return self._rng.choice(best)


__all__ = ["MinimaxAgent", "MoveOrder", "PolicyOrder", "captures_first"]
