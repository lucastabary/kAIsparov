from __future__ import annotations

import torch
from torch_geometric.data import Data

from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType
from kaisparov.core.utils import coord_to_index, get_piece_value, index_to_coord
from kaisparov.models.base_processor import (
    BaseProcessor,
    ModelAction,
    create_static_full_chess_graph,
)
from kaisparov.models.base_processor import (
    get_legal_mask as base_get_legal_mask,
)
from kaisparov.training.ppo import PPOBuffer, train_one_epoch


def compute_reward(game: ChessGame, captured_piece: Piece | None) -> float:
    reward = 0.0
    if captured_piece is not None:
        reward += get_piece_value(captured_piece.type)

    # TODO(phase 4): add shaping (e.g. bonus when the opponent is in check).
    return reward


class GNN1Processor(BaseProcessor):
    def __init__(self):
        self.static_graph_edges = create_static_full_chess_graph()

    def graphify(self, game: ChessGame) -> Data:
        piece_order = [
            PieceType.KING,
            PieceType.QUEEN,
            PieceType.BISHOP,
            PieceType.ROOK,
            PieceType.KNIGHT,
            PieceType.PAWN,
        ]
        piece_to_idx = {pt: i for i, pt in enumerate(piece_order)}

        current_player = game.turn
        x = torch.zeros((BOARD_SIZE * BOARD_SIZE, 12), dtype=torch.float32)

        for col in range(BOARD_SIZE):
            for row in range(BOARD_SIZE):
                piece = game.grid[col][row]
                if piece is None:
                    continue

                node_idx = coord_to_index((col, row))
                piece_idx = piece_to_idx[piece.type]
                if piece.player == current_player:
                    x[node_idx, piece_idx] = 1.0
                else:
                    x[node_idx, 6 + piece_idx] = 1.0

        static_edge_index, static_edge_type = self.static_graph_edges
        return Data(x=x, edge_index=static_edge_index, edge_type=static_edge_type)

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
            legal_mask = get_legal_mask(game, edge_index)
        if not legal_mask.any():
            raise RuntimeError("No legal action available for current game state.")

        masked_logits = action_scores.masked_fill(~legal_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_logits)

        action = torch.argmax(masked_logits) if deterministic else dist.sample()

        action_index = int(action.item())
        source_idx = int(edge_index[0, action_index].item())
        dest_idx = int(edge_index[1, action_index].item())
        source = index_to_coord(source_idx)
        dest = index_to_coord(dest_idx)

        return ModelAction(
            move_coords=((int(source[0]), int(source[1])), (int(dest[0]), int(dest[1]))),
            action_index=action_index,
            log_prob=dist.log_prob(action),
            value=value.squeeze(),
            entropy=dist.entropy(),
        )


def _coord_to_index_adapter(coord: tuple[int, int], board_size: int) -> int:
    _ = board_size
    return coord_to_index(coord)


def get_legal_mask(game: ChessGame, edge_index: torch.Tensor) -> torch.Tensor:
    return base_get_legal_mask(game, edge_index, coord_to_index_fn=_coord_to_index_adapter)


__all__ = [
    "GNN1Processor",
    "PPOBuffer",
    "ModelAction",
    "coord_to_index",
    "index_to_coord",
    "get_legal_mask",
    "train_one_epoch",
    "compute_reward",
]
