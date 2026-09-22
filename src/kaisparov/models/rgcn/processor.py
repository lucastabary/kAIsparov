from __future__ import annotations

import torch
from torch_geometric.data import Data

from kaisparov.core.game import ChessGame
from kaisparov.core.pieces import BOARD_SIZE
from kaisparov.core.utils import coord_to_index, index_to_coord
from kaisparov.models.base_processor import (
    BaseProcessor,
    ModelAction,
    aggregate_edge_logits_to_moves,
)
from kaisparov.models.features import DEFAULT_FEATURES, get_feature_set
from kaisparov.models.rgcn.graph import create_static_full_chess_graph

NUM_NODES = BOARD_SIZE * BOARD_SIZE


class RGCNProcessor(BaseProcessor):
    """Board <-> graph for the RGCN backends.

    The graph is this backend's: the static edge set (one relation per movement
    type) and the edge-to-move decoding below. What sits on the nodes is not — it is
    a named :mod:`~kaisparov.models.features` set, which must be the one the model
    was built for (``model.features``).
    """

    def __init__(self, features: str = DEFAULT_FEATURES):
        self.features = get_feature_set(features)
        self.static_graph_edges = create_static_full_chess_graph()

    def graphify(self, game: ChessGame) -> Data:
        edge_index, edge_type = self.static_graph_edges
        x = torch.from_numpy(self.features.encode(game))
        return Data(x=x, edge_index=edge_index, edge_type=edge_type)

    def graphify_batch(self, games: list[ChessGame]) -> list[Data]:
        """Vectorised :meth:`graphify` for several games — identical output, faster.

        This is the training hot path: the rollout calls it once per ply for every
        game still running, and the feature set encodes the whole batch at once.
        """
        if not games:
            return []
        x = torch.from_numpy(self.features.encode_batch(games))
        edge_index, edge_type = self.static_graph_edges
        # Clone each row so a stored state does not keep the whole batch alive.
        return [
            Data(x=x[i].clone(), edge_index=edge_index, edge_type=edge_type)
            for i in range(len(games))
        ]

    def process_output(
        self,
        model_output: tuple[torch.Tensor, torch.Tensor],
        game: ChessGame,
        deterministic: bool,
        legal_mask: torch.Tensor | None = None,
    ) -> ModelAction:
        action_scores, value = model_output
        edge_index = self.static_graph_edges[0].to(action_scores.device)

        if legal_mask is None:
            legal_mask = self.legal_mask(game, edge_index)
        if not legal_mask.any():
            raise RuntimeError("No legal action available for current game state.")

        # Distribution over MOVES, not edges: several edges can denote the same
        # (src, dst) move, and a per-edge argmax fragments a move's probability across
        # its edges (a king step spans king+rook/bishop+queen edges, a knight jump is
        # one), biasing greedy play against king moves. Aggregate first.
        num_nodes = NUM_NODES
        move_keys, move_logits = aggregate_edge_logits_to_moves(
            action_scores, edge_index, legal_mask, num_nodes
        )
        dist = torch.distributions.Categorical(logits=move_logits)

        move_pos = torch.argmax(move_logits) if deterministic else dist.sample()
        move_key = int(move_keys[move_pos].item())
        source = index_to_coord(move_key // num_nodes)
        dest = index_to_coord(move_key % num_nodes)

        return ModelAction(
            move_coords=((int(source[0]), int(source[1])), (int(dest[0]), int(dest[1]))),
            action_index=move_key,  # now a MOVE key (src*num_nodes + dst), not an edge index
            log_prob=dist.log_prob(move_pos),
            value=value.squeeze(),
            entropy=dist.entropy(),
        )

    def legal_mask(self, game: ChessGame, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        """Which edges are legal moves right now (see :func:`get_legal_mask`).

        ``edge_index`` is this processor's static graph, already moved to the model's
        device by a caller that runs the mask once per ply; ``None`` uses it as built.
        """
        return get_legal_mask(
            game, self.static_graph_edges[0] if edge_index is None else edge_index
        )

    def move_mask(self, game: ChessGame, moves) -> torch.Tensor:
        """Boolean edge mask covering exactly ``moves`` (a list of ``(src, dst)`` coords).

        Same edge-packing as :func:`get_legal_mask`, so the result can be passed as
        ``legal_mask`` to :meth:`process_output` to restrict the policy to an explicit
        move set (e.g. the king-safe moves — see ``NeuralAgent(avoid_king_suicide=...)``).
        """
        edge_index = self.static_graph_edges[0]
        num_nodes = BOARD_SIZE * BOARD_SIZE
        if not moves:
            return torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        keys = [coord_to_index(m[0]) * num_nodes + coord_to_index(m[1]) for m in moves]
        return _mask_from_keys(keys, edge_index, num_nodes)


# edge_index -> its packed (src * 64 + dst) keys. The static graph is built once per
# processor and never changes, but the mask is rebuilt every ply of every game, and
# packing 4096 edges again each time costs more than the mask itself. Keyed by id()
# with the tensor kept alive alongside, so an id is never reused for another graph.
_PACKED_EDGES: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _packed_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    cached = _PACKED_EDGES.get(id(edge_index))
    if cached is None:
        packed = edge_index[0] * num_nodes + edge_index[1]
        _PACKED_EDGES[id(edge_index)] = (edge_index, packed)
        return packed
    return cached[1]


# One row per possible (src, dst) pair, reused across calls: writing the wanted keys
# into it and reading it back at the edges is a couple of gathers, where torch.isin
# sorts both sides and cost more than generating the move list in the first place.
_KEY_TABLE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _mask_from_keys(keys: list[int], edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Which edges carry one of these packed ``src * 64 + dst`` keys."""
    device = edge_index.device
    if not keys:
        return torch.zeros(edge_index.shape[1], dtype=torch.bool, device=device)
    table = _KEY_TABLE.get((num_nodes, device))
    if table is None:
        table = torch.zeros(num_nodes * num_nodes, dtype=torch.bool, device=device)
        _KEY_TABLE[(num_nodes, device)] = table
    table.zero_()
    table[torch.tensor(keys, dtype=torch.long, device=device)] = True
    return table[_packed_edges(edge_index, num_nodes)]


def get_legal_mask(game: ChessGame, edge_index: torch.Tensor) -> torch.Tensor:
    """Boolean mask over graph edges: which ``(source, dest)`` pairs are legal now.

    Reads python-chess's move list directly rather than going through
    :class:`~kaisparov.core.move.Move`: its square indices are already this project's
    ``row * 8 + col``, so the edge key is ``from_square * 64 + to_square`` with no
    conversion at all. This runs once per game per ply during training.

    The four promotions of one pawn push share a ``(source, dest)`` pair and so a
    single edge; playing it queens. Underpromotion is not in the action space.
    """
    num_nodes = NUM_NODES
    keys = {move.from_square * num_nodes + move.to_square for move in game.board.legal_moves}
    return _mask_from_keys(list(keys), edge_index, num_nodes)


__all__ = ["RGCNProcessor", "get_legal_mask"]
