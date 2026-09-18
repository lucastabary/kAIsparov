from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from kaisparov.core import bitboard_batch as bbb
from kaisparov.core.game import ChessGame, Undo
from kaisparov.core.pieces import BOARD_SIZE, PieceType, Player
from kaisparov.core.rules import attacked_squares
from kaisparov.core.utils import coord_to_index, get_piece_value, index_to_coord
from kaisparov.models.base_processor import (
    BaseProcessor,
    ModelAction,
    aggregate_edge_logits_to_moves,
    create_static_full_chess_graph,
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
# chess.Board bitboard attributes, in the same order as _PIECE_ORDER above, so the
# one-hot feature index of a type is its index here.
_CHESS_BITBOARDS = ("kings", "queens", "bishops", "rooks", "knights", "pawns")


def compute_reward(game: ChessGame, undo: Undo) -> float:
    """Fallback reward when the trainer supplies none: material only.

    Mirrors :class:`~kaisparov.envs.chess_env.ChessEnv` — the captured piece, plus
    what a promotion gained. Shaping (checkmate bonus, check, step penalty) lives in
    :mod:`kaisparov.training.reward`.
    """
    reward = 0.0
    if undo.captured is not None:
        reward += get_piece_value(undo.captured.type)
    if undo.move.promotion is not None:
        reward += get_piece_value(undo.move.promotion) - get_piece_value(PieceType.PAWN)
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
        for cx, cy in attacked_squares(game, enemy_player):
            x[coord_to_index((cx, cy)), 12] = 1.0
        for cx, cy in attacked_squares(game, current_player):
            x[coord_to_index((cx, cy)), 13] = 1.0

        static_edge_index, static_edge_type = self.static_graph_edges
        return Data(x=x, edge_index=static_edge_index, edge_type=static_edge_type)

    def graphify_batch(self, games: list[ChessGame]) -> list[Data]:
        """Vectorised :meth:`graphify` for several games — identical output, faster.

        Nothing here walks the board. python-chess already holds each position as one
        bitboard per piece type plus a per-colour occupancy, in this project's own
        ``sq = row * 8 + col`` convention, so the twelve one-hot piece planes unpack
        straight out of those masks with numpy shifts, and the two blocking-aware
        control flags come from the vectorised Kogge-Stone maps in
        :mod:`kaisparov.core.bitboard_batch` for the whole batch at once.

        This is the training hot path: the rollout calls it once per ply for every
        game still running.
        """
        n = len(games)
        if n == 0:
            return []

        num_nodes = BOARD_SIZE * BOARD_SIZE
        boards = [game.board for game in games]
        white_to_move = np.fromiter((b.turn for b in boards), dtype=bool, count=n)

        # (6, n) masks for the side to move and for its opponent, in _PIECE_ORDER.
        own = np.zeros((6, n), dtype=np.uint64)
        enemy = np.zeros((6, n), dtype=np.uint64)
        for i, board in enumerate(boards):
            mine = board.occupied_co[board.turn]
            theirs = board.occupied_co[not board.turn]
            for k, attribute in enumerate(_CHESS_BITBOARDS):
                bb = getattr(board, attribute)
                own[k, i] = bb & mine
                enemy[k, i] = bb & theirs

        one = np.uint64(1)
        x = torch.zeros((n, num_nodes, 14), dtype=torch.float32)
        for k in range(6):
            # Features 0-5: the side to move's pieces. 6-11: the opponent's.
            x[:, :, k] = torch.from_numpy(((own[k][:, None] >> _SQUARES) & one).astype(np.float32))
            x[:, :, 6 + k] = torch.from_numpy(
                ((enemy[k][:, None] >> _SQUARES) & one).astype(np.float32)
            )

        atk_white = bbb.attacked_by_packed(bbb.pack_boards(boards, bbb.WHITE), bbb.WHITE)
        atk_black = bbb.attacked_by_packed(bbb.pack_boards(boards, bbb.BLACK), bbb.BLACK)
        # Feature 12 = attacked by the opponent (side not to move); 13 = by the side to move.
        atk_enemy = np.where(white_to_move, atk_black, atk_white)
        atk_current = np.where(white_to_move, atk_white, atk_black)
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
        keys = torch.tensor(
            [coord_to_index(m[0]) * num_nodes + coord_to_index(m[1]) for m in moves],
            dtype=edge_index.dtype,
            device=edge_index.device,
        )
        return torch.isin(_packed_edges(edge_index, num_nodes), keys)


def _coord_to_index_adapter(coord: tuple[int, int], board_size: int) -> int:
    _ = board_size
    return coord_to_index(coord)


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


def get_legal_mask(game: ChessGame, edge_index: torch.Tensor) -> torch.Tensor:
    """Boolean mask over graph edges: which ``(source, dest)`` pairs are legal now.

    Reads python-chess's move list directly rather than going through
    :class:`~kaisparov.core.move.Move`: its square indices are already this project's
    ``row * 8 + col``, so the edge key is ``from_square * 64 + to_square`` with no
    conversion at all. This runs once per game per ply during training.

    The four promotions of one pawn push share a ``(source, dest)`` pair and so a
    single edge; playing it queens. Underpromotion is not in the action space.
    """
    num_nodes = BOARD_SIZE * BOARD_SIZE
    keys = {move.from_square * num_nodes + move.to_square for move in game.board.legal_moves}
    if not keys:
        return torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
    key_tensor = torch.tensor(sorted(keys), dtype=edge_index.dtype, device=edge_index.device)
    return torch.isin(_packed_edges(edge_index, num_nodes), key_tensor)


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
