from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from kaisparov.core import bitboard_batch as bbb
from kaisparov.core.board import ChessGame
from kaisparov.core.pieces import BOARD_SIZE, Piece, PieceType, Player
from kaisparov.core.rules import attacked_squares
from kaisparov.core.utils import coord_to_index, get_piece_value, index_to_coord
from kaisparov.models.base_processor import (
    BaseProcessor,
    ModelAction,
    aggregate_edge_logits_to_moves,
    create_static_full_chess_graph,
)
from kaisparov.models.base_processor import (
    get_legal_mask as base_get_legal_mask,
)
from kaisparov.training.ppo import PPOBuffer, train_one_epoch

# Ally/enemy one-hot slot per piece type (features 0-5 ally, 6-11 enemy).
_PIECE_ORDER = (
    PieceType.KING,
    PieceType.QUEEN,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.KNIGHT,
    PieceType.PAWN,
)
_PIECE_IDX = {pt: i for i, pt in enumerate(_PIECE_ORDER)}
# Which packed bitboard rows (orth, diag, knights, kings, pawns) a type contributes
# to; the queen is both an orthogonal and a diagonal slider.
_BB_ROWS = {
    PieceType.QUEEN: (bbb.ORTH, bbb.DIAG),
    PieceType.ROOK: (bbb.ORTH,),
    PieceType.BISHOP: (bbb.DIAG,),
    PieceType.KNIGHT: (bbb.KNIGHTS,),
    PieceType.KING: (bbb.KINGS,),
    PieceType.PAWN: (bbb.PAWNS,),
}
_SQUARES = np.arange(BOARD_SIZE * BOARD_SIZE, dtype=np.uint64)


def compute_reward(game: ChessGame, captured_piece: Piece | None) -> float:
    reward = 0.0
    if captured_piece is not None:
        reward += get_piece_value(captured_piece.type)

    # TODO(phase 4): add shaping (e.g. bonus when the opponent is in check).
    return reward


class RGCNProcessor(BaseProcessor):
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
        enemy_player = Player.BLACK if current_player == Player.WHITE else Player.WHITE

        # 14 features/node: 6 ally piece-type one-hot, 6 enemy piece-type one-hot,
        # then two position-aware, blocking-aware control flags (see below). Without
        # the control flags a static geometric graph cannot tell a real attack from a
        # blocked line, so "my king is in check" is not perceivable and the policy
        # learns to attack but never to defend the king. These flags hand that
        # blocking-aware reasoning to the engine, which already knows it.
        x = torch.zeros((BOARD_SIZE * BOARD_SIZE, 14), dtype=torch.float32)

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

        # Feature 12: attacked by the opponent (side NOT to move) — set on the ally
        # king's square exactly when it is in check, and on empty squares that are
        # unsafe to move onto. Feature 13: defended by the side to move.
        for cx, cy in attacked_squares(game.grid, enemy_player):
            x[coord_to_index((cx, cy)), 12] = 1.0
        for cx, cy in attacked_squares(game.grid, current_player):
            x[coord_to_index((cx, cy)), 13] = 1.0

        static_edge_index, static_edge_type = self.static_graph_edges
        return Data(x=x, edge_index=static_edge_index, edge_type=static_edge_type)

    def graphify_batch(self, games: list[ChessGame]) -> list[Data]:
        """Vectorised :meth:`graphify` for several games — identical output, faster.

        The one-hot piece features (0-11) are filled in a single per-square scan per
        game (which also packs that game's bitboards for free); the two blocking-aware
        control flags (12-13) are then computed for *all* games at once with the
        vectorised Kogge-Stone maps in :mod:`kaisparov.core.bitboard_batch`, instead of
        two per-game :func:`attacked_squares` calls. Feature index == square index
        (``row*8+col``), so the control bitboards unpack straight into the columns.
        """
        n = len(games)
        if n == 0:
            return []

        num_nodes = BOARD_SIZE * BOARD_SIZE
        x = torch.zeros((n, num_nodes, 14), dtype=torch.float32)
        packed_white = np.zeros((6, n), dtype=np.uint64)
        packed_black = np.zeros((6, n), dtype=np.uint64)
        white_to_move = np.zeros(n, dtype=bool)

        for i, game in enumerate(games):
            current = game.turn
            white_to_move[i] = current == Player.WHITE
            grid = game.grid
            xi = x[i]
            occ = 0
            white_rows = [0, 0, 0, 0, 0]
            black_rows = [0, 0, 0, 0, 0]

            for col in range(BOARD_SIZE):
                column = grid[col]
                for row in range(BOARD_SIZE):
                    piece = column[row]
                    if piece is None:
                        continue
                    sq = row * BOARD_SIZE + col  # == coord_to_index((col, row)) == node index
                    piece_idx = _PIECE_IDX[piece.type]
                    if piece.player == current:
                        xi[sq, piece_idx] = 1.0
                    else:
                        xi[sq, 6 + piece_idx] = 1.0
                    bit = 1 << sq
                    occ |= bit
                    rows = white_rows if piece.player == Player.WHITE else black_rows
                    for r in _BB_ROWS[piece.type]:
                        rows[r] |= bit

            for r in range(5):
                packed_white[r, i] = white_rows[r]
                packed_black[r, i] = black_rows[r]
            packed_white[bbb.OCC, i] = occ
            packed_black[bbb.OCC, i] = occ

        atk_white = bbb.attacked_by_packed(packed_white, bbb.WHITE)
        atk_black = bbb.attacked_by_packed(packed_black, bbb.BLACK)
        # Feature 12 = attacked by the opponent (side not to move); 13 = by the side to move.
        atk_enemy = np.where(white_to_move, atk_black, atk_white)
        atk_current = np.where(white_to_move, atk_white, atk_black)
        one = np.uint64(1)
        x[:, :, 12] = torch.from_numpy(((atk_enemy[:, None] >> _SQUARES) & one).astype(np.float32))
        x[:, :, 13] = torch.from_numpy(
            ((atk_current[:, None] >> _SQUARES) & one).astype(np.float32)
        )

        edge_index, edge_type = self.static_graph_edges
        return [Data(x=x[i].clone(), edge_index=edge_index, edge_type=edge_type) for i in range(n)]

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

        # Distribution over MOVES, not edges: several edges can denote the same
        # (src, dst) move, and a per-edge argmax fragments a move's probability across
        # its edges (a king step spans king+rook/bishop+queen edges, a knight jump is
        # one), biasing greedy play against king moves. Aggregate first.
        num_nodes = len(game.grid) ** 2
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


def _coord_to_index_adapter(coord: tuple[int, int], board_size: int) -> int:
    _ = board_size
    return coord_to_index(coord)


def get_legal_mask(game: ChessGame, edge_index: torch.Tensor) -> torch.Tensor:
    return base_get_legal_mask(game, edge_index, coord_to_index_fn=_coord_to_index_adapter)


__all__ = [
    "RGCNProcessor",
    "PPOBuffer",
    "ModelAction",
    "coord_to_index",
    "index_to_coord",
    "get_legal_mask",
    "train_one_epoch",
    "compute_reward",
]
