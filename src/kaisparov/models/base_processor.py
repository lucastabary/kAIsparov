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
