"""The static graph ``rgcn`` and ``shared_rgcn`` reason over.

One edge per way a piece could move from one square to another *on an empty board*,
typed by the movement (``edge_type``): 0 knight, 1 rook, 2 bishop, 3 king, 4 white
pawn, 5 black pawn. The queen has no relation of its own; its moves are the union of
the rook's and the bishop's. There is no castling edge, so the policy cannot castle.

The same graph for every position: what differs between positions is carried by the
node features (``models/features.py``) and by which edges are legal (the mask).
"""

from __future__ import annotations

import torch

NUM_RELATIONS = 6


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
