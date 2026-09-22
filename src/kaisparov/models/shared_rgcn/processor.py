"""Board <-> graph encoding for ``shared_rgcn``.

The encoding is *exactly* the one ``rgcn`` uses — same static edge set, same node
feature sets, same edge-to-move decoding — because only the network differs between
the two backends. Reusing ``RGCNProcessor`` (rather than copying it) is what keeps
the comparison between them honest: any change to the representation applies to both.
"""

from __future__ import annotations

from kaisparov.models.base_processor import ModelAction
from kaisparov.models.rgcn.processor import (
    RGCNProcessor,
    compute_reward,
    get_legal_mask,
)
from kaisparov.training.ppo import PPOBuffer, train_one_epoch


class SharedRGCNProcessor(RGCNProcessor):
    """``RGCNProcessor`` under this backend's name — identical behaviour."""


__all__ = [
    "SharedRGCNProcessor",
    "PPOBuffer",
    "ModelAction",
    "get_legal_mask",
    "train_one_epoch",
    "compute_reward",
]
