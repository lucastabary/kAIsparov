from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class ModelAction:
    move_coords: tuple[tuple[int, int], tuple[int, int]]
    action_index: int
    log_prob: Tensor
    value: Tensor
    entropy: Tensor


class BaseProcessor(ABC):
    """Interface for board -> model-input converters. One per model backend."""

    @abstractmethod
    def graphify(self, game):
        """Convert a ChessGame into a model-ready object (e.g. a PyG ``Data``)."""

    def graphify_batch(self, games):
        """Convert several games at once. Default: one :meth:`graphify` per game.

        Backends can override with a vectorised implementation (see the RGCN
        processor); callers that graphify a whole epoch's active games at once
        (the self-play rollout) should prefer this over a per-game loop.
        """
        return [self.graphify(game) for game in games]


def default_coord_to_index(coord: tuple[int, int], board_size: int) -> int:
    return coord[1] * board_size + coord[0]


def get_legal_mask(
    game, edge_index: torch.Tensor, coord_to_index_fn=default_coord_to_index
) -> torch.Tensor:
    """Build a boolean mask for legal edges for the current player.

    Args:
        game: Game object exposing `grid`, `turn` and `possible_moves`.
        edge_index: Tensor of shape [2, E].
        coord_to_index_fn: Callable mapping (x, y) to node index.
    """
    board_size = len(game.grid)
    num_nodes = board_size * board_size

    legal_pairs: list[int] = []

    for col in range(board_size):
        for row in range(board_size):
            piece = game.grid[col][row]
            if piece is None or piece.player != game.turn:
                continue

            source_idx = coord_to_index_fn((col, row), board_size)
            for dest in game.possible_moves((col, row)):
                dest_idx = coord_to_index_fn(dest, board_size)
                legal_pairs.append(source_idx * num_nodes + dest_idx)

    if not legal_pairs:
        return torch.zeros(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

    legal_pairs_tensor = torch.tensor(legal_pairs, dtype=edge_index.dtype, device=edge_index.device)
    packed_edges = edge_index[0] * num_nodes + edge_index[1]
    mask = torch.isin(packed_edges, legal_pairs_tensor)

    return mask


def aggregate_edge_logits_to_moves(
    edge_logits: Tensor, edge_index: Tensor, legal_mask: Tensor, num_nodes: int = 64
) -> tuple[Tensor, Tensor]:
    """Collapse per-edge logits into per-MOVE logits.

    Several graph edges can denote the same ``(source, dest)`` move: a one-square king
    step is *also* a 1-square rook/bishop/queen edge, so it appears as up to 3 edges,
    while a knight jump is a single edge. The policy must be a distribution over
    *moves*, not edges -- otherwise a move spread over ``k`` edges has its probability
    fragmented, and a greedy ``argmax`` over edges is biased toward single-edge moves
    (knights) over multi-edge ones (the king), suppressing exactly the king moves that
    save it from check.

    We combine the edges of one move with ``logsumexp``, which makes
    ``softmax(move_logits)[m]`` equal the summed edge probabilities of move ``m`` under
    the original per-edge softmax. Only the legal edges take part.

    Returns ``(move_keys, move_logits)`` where ``move_key = src_idx * num_nodes +
    dst_idx`` uniquely identifies a move and is stable across recomputation (so PPO can
    match a stored action to its logit).
    """
    legal_idx = legal_mask.nonzero(as_tuple=True)[0]
    keys = edge_index[0, legal_idx] * num_nodes + edge_index[1, legal_idx]
    logits = edge_logits[legal_idx]

    move_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    m = int(move_keys.numel())

    # Stable per-group logsumexp: subtract each group's max before exp/sum.
    max_per = torch.full((m,), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_per = max_per.scatter_reduce(0, inverse, logits, reduce="amax", include_self=True)
    shifted = torch.exp(logits - max_per.index_select(0, inverse))
    sum_per = torch.zeros(m, device=logits.device, dtype=logits.dtype).index_add(
        0, inverse, shifted
    )
    move_logits = max_per + torch.log(sum_per)
    return move_keys, move_logits


def create_static_full_chess_graph():
    edge_index = []
    edge_type = []

    # Mapping des types d'arêtes (edge_attr)
    # 0: Cavalier, 1: Tour, 2: Fou, 3: Roi, 4: Pion Blanc, 5: Pion Noir

    def is_on_board(r, c):
        return 0 <= r < 8 and 0 <= c < 8

    def get_idx(r, c):
        return r * 8 + c

    for r in range(8):
        for c in range(8):
            curr = get_idx(r, c)

            # --- 1. CAVALIER (Knight) ---
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in knight_moves:
                if is_on_board(r + dr, c + dc):
                    edge_index.append([curr, get_idx(r + dr, c + dc)])
                    edge_type.append(0)

            # --- 2. TOUR (Rook) & REINE (partiel) ---
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for dr, dc in directions:
                for dist in range(1, 8):
                    if is_on_board(r + dr * dist, c + dc * dist):
                        edge_index.append([curr, get_idx(r + dr * dist, c + dc * dist)])
                        edge_type.append(1)
                    else:
                        break

            # --- 3. FOU (Bishop) & REINE (partiel) ---
            diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr, dc in diagonals:
                for dist in range(1, 8):
                    if is_on_board(r + dr * dist, c + dc * dist):
                        edge_index.append([curr, get_idx(r + dr * dist, c + dc * dist)])
                        edge_type.append(2)
                    else:
                        break

            # --- 4. ROI (King) ---
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    if is_on_board(r + dr, c + dc):
                        edge_index.append([curr, get_idx(r + dr, c + dc)])
                        edge_type.append(3)

            # --- 5. PION BLANC (White Pawn) ---
            # Avancée
            if is_on_board(r + 1, c):
                edge_index.append([curr, get_idx(r + 1, c)])
                edge_type.append(4)
                if r == 1 and is_on_board(r + 2, c):  # Double poussée initiale
                    edge_index.append([curr, get_idx(r + 2, c)])
                    edge_type.append(4)
            # Captures
            for dc in [-1, 1]:
                if is_on_board(r + 1, c + dc):
                    edge_index.append([curr, get_idx(r + 1, c + dc)])
                    edge_type.append(4)

            # --- 6. PION NOIR (Black Pawn) ---
            if is_on_board(r - 1, c):
                edge_index.append([curr, get_idx(r - 1, c)])
                edge_type.append(5)
                if r == 6 and is_on_board(r - 2, c):
                    edge_index.append([curr, get_idx(r - 2, c)])
                    edge_type.append(5)
            for dc in [-1, 1]:
                if is_on_board(r - 1, c + dc):
                    edge_index.append([curr, get_idx(r - 1, c + dc)])
                    edge_type.append(5)

            # --- 7. ROQUE ---

    # Conversion en tenseurs PyG [2, E] et [E]
    edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type_t = torch.tensor(edge_type, dtype=torch.long)

    return edge_index_t, edge_type_t
