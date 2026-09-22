"""What it takes to rebuild a model: the one record every loader goes through.

A checkpoint is a bare ``state_dict``; the weights do not say which network they
belong to. :class:`Architecture` is the rest of the answer — the backend, its width,
and the node feature set it reads — and a run records every field of it in its
config, so a run can always be rebuilt from its own folder.

To add a hyper-parameter that changes the network's shape or its inputs, add it
here, to ``TrainConfig`` and to :func:`kaisparov.models.factory.build_agent` — not as
one more keyword threaded through each place that builds a model.

Torch-free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from kaisparov.models.features import DEFAULT_FEATURES, get_feature_set

DEFAULT_MODEL = "rgcn"


@dataclass(frozen=True)
class Architecture:
    model: str = DEFAULT_MODEL
    hidden_dim: int = 8
    features: str = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        get_feature_set(self.features)  # fail on an unknown name, not deep in a forward

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.model} (hidden_dim={self.hidden_dim}, features={self.features})"


__all__ = ["Architecture", "DEFAULT_MODEL"]
