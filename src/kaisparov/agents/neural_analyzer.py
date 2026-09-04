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

from kaisparov.core.utils import index_to_coord
from kaisparov.insights import MoveInsight, PositionAnalysis
from kaisparov.models.base_processor import default_coord_to_index, get_legal_mask


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
            legal_mask = get_legal_mask(game, edge_index, default_coord_to_index)
            if not legal_mask.any():
                return None

            masked = action_scores.masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(masked, dim=0)

            # The static graph carries several typed edges for one (src, dst) square
            # pair (e.g. a rook step, a king step and a pawn push coincide), so sum
            # each move's probability mass across its edges to rank distinct moves.
            move_prob: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
            legal_idx = legal_mask.nonzero(as_tuple=False).flatten().tolist()
            for idx in legal_idx:
                source = index_to_coord(int(edge_index[0, idx]))
                dest = index_to_coord(int(edge_index[1, idx]))
                move = ((int(source[0]), int(source[1])), (int(dest[0]), int(dest[1])))
                move_prob[move] = move_prob.get(move, 0.0) + float(probs[idx])

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
