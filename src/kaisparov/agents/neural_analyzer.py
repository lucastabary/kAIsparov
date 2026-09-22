"""Turn a trained actor-critic into a :class:`~kaisparov.insights.Analyzer`.

Where :class:`~kaisparov.agents.neural_agent.NeuralAgent` only returns *the* move,
this exposes the reasoning behind it: the critic's value for the side to move and
the top-``k`` legal moves ranked by the actor's probability. That is exactly the
material the developer-mode overlay needs to highlight "interesting" squares and
move ideas during play.

It shares a model + processor with the agent (build both from one loaded model),
so producing an analysis costs a single extra forward pass per position.
"""

from __future__ import annotations

import torch

from kaisparov.core.move import Move
from kaisparov.core.pieces import BOARD_SIZE
from kaisparov.core.utils import index_to_coord
from kaisparov.insights import MoveInsight, PositionAnalysis
from kaisparov.models.base_processor import aggregate_edge_logits_to_moves

NUM_NODES = BOARD_SIZE * BOARD_SIZE


class NeuralAnalyzer:
    name = "neural"

    def __init__(self, model: torch.nn.Module, processor, top_k: int = 4):
        self.model = model
        self.processor = processor
        self.top_k = top_k

    def _device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def analyze(self, game) -> PositionAnalysis | None:
        self.model.eval()
        with torch.no_grad():
            data = self.processor.graphify(game).to(self._device())
            action_scores, value = self.model(data)

            edge_index = self.processor.static_graph_edges[0].to(action_scores.device)
            legal_mask = self.processor.legal_mask(game, edge_index)
            if not legal_mask.any():
                return None

            # Several typed edges can carry one (src, dst) move; the policy's
            # distribution is over moves, as in process_output, so rank moves by it.
            move_keys, move_logits = aggregate_edge_logits_to_moves(
                action_scores, edge_index, legal_mask, NUM_NODES
            )
            probs = torch.softmax(move_logits, dim=0)
            move_prob: dict[Move, float] = {}
            for key, prob in zip(move_keys.tolist(), probs.tolist(), strict=True):
                source = index_to_coord(key // NUM_NODES)
                dest = index_to_coord(key % NUM_NODES)
                move_prob[Move((int(source[0]), int(source[1])), (int(dest[0]), int(dest[1])))] = (
                    prob
                )

            ranked = sorted(move_prob.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k]
            best = ranked[0][1] or 1.0  # guard against an all-zero degenerate softmax

            candidates = [
                MoveInsight(move=move, score=prob / best, label=f"{prob * 100:.0f}%")
                for move, prob in ranked
            ]

            return PositionAnalysis(
                candidates=tuple(candidates),
                value=float(value.reshape(-1)[0]),
                source=self.name,
            )
