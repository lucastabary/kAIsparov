"""Dynamic backend loading by name, and the one way to build or load an agent.

Every place that needs a network — the trainer, its opponent pool and workers,
``play``, ``eval``, ``bench`` — goes through :func:`build_agent` (fresh weights) or
:func:`load_agent` (a checkpoint). That is what keeps a model and its processor
built for the same :class:`~kaisparov.models.architecture.Architecture`.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kaisparov.models.architecture import DEFAULT_MODEL, Architecture
from kaisparov.models.backend_spec import BackendSpec
from kaisparov.models.base_model import BaseModel
from kaisparov.models.features import DEFAULT_FEATURES, feature_set_for_dim

MODEL_MODULES = {
    "rgcn": "kaisparov.models.rgcn",
    "shared_rgcn": "kaisparov.models.shared_rgcn",
}


def resolve_model_name(model_name: str | None = None) -> str:
    return model_name or DEFAULT_MODEL


def load_backend(model_name: str | None = None):
    resolved = resolve_model_name(model_name)
    return importlib.import_module(MODEL_MODULES.get(resolved, resolved))


def load_backend_spec(model_name: str | None = None) -> BackendSpec:
    backend = load_backend(model_name)
    spec = getattr(backend, "BACKEND_SPEC", None)
    if spec is None:
        raise AttributeError(f"Model backend '{backend.__name__}' must expose BACKEND_SPEC.")
    if not isinstance(spec, BackendSpec):
        raise TypeError(f"BACKEND_SPEC in '{backend.__name__}' must be a BackendSpec.")
    return spec


# ------------------------------------------------------------ reading weights
# state_dict keys whose first dimension equals hidden_dim, tried in order.
_HIDDEN_DIM_KEYS = ("chess_rgcn.conv1.bias", "critic_head.0.bias", "actor_head.0.bias")
# (key, axis) where the input layer's width — the feature set's dim — can be read.
_INPUT_DIM_KEYS = (("chess_rgcn.conv1.root", 0), ("chess_rgcn.encoder.weight", 1))


def infer_hidden_dim(state_dict) -> int | None:
    """Read hidden_dim off a checkpoint whose run does not record it.

    Every candidate key is a 1-D tensor of length ``hidden_dim``, so its shape tells
    us the width the model was trained at.
    """
    for key in _HIDDEN_DIM_KEYS:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.dim() >= 1:
            return int(tensor.shape[0])
    return None


def infer_features(state_dict) -> str | None:
    """The feature set a checkpoint was trained on, read off its input layer's width.

    For runs that predate the ``features`` key. Exact as long as every feature set
    has its own width (see :func:`~kaisparov.models.features.feature_set_for_dim`).
    """
    for key, axis in _INPUT_DIM_KEYS:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.dim() > axis:
            return feature_set_for_dim(int(tensor.shape[axis]))
    return None


def run_record(checkpoint: str | Path) -> dict[str, Any]:
    """The ``run.json`` of the run a checkpoint belongs to, or ``{}`` if untracked.

    Tracked checkpoints live at ``runs/<id>/checkpoints/<file>.pth``.
    """
    path = Path(checkpoint).parent.parent / "run.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def resolve_architecture(
    checkpoint: str | Path,
    state_dict=None,
    *,
    model: str | None = None,
    hidden_dim: int | None = None,
    features: str | None = None,
) -> Architecture:
    """The architecture of ``checkpoint``, each field from the best source available.

    An explicit argument wins; then what the checkpoint's run recorded; then what the
    weights themselves show (older runs did not record ``features``, and a raw file
    records nothing); then the defaults. ``state_dict`` is read from ``checkpoint``
    only if the weights have to be looked at.
    """
    record = run_record(checkpoint)
    config = record.get("config") or {}
    model = model or record.get("model") or config.get("model") or DEFAULT_MODEL
    hidden_dim = hidden_dim or config.get("hidden_dim")
    features = features or config.get("features")
    if hidden_dim is None or features is None:
        if state_dict is None:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        hidden_dim = hidden_dim or infer_hidden_dim(state_dict) or 8
        features = features or infer_features(state_dict) or DEFAULT_FEATURES
    return Architecture(model=model, hidden_dim=int(hidden_dim), features=features)


# --------------------------------------------------------------- building
def build_agent(architecture: Architecture, device: torch.device) -> tuple[BaseModel, Any]:
    """A freshly initialised ``(model, processor)`` pair for ``architecture``."""
    spec = load_backend_spec(architecture.model)
    model = spec.model_class.create_agent(
        device=device, hidden_dim=architecture.hidden_dim, features=architecture.features
    )
    return model, spec.processor_class(features=architecture.features)


@dataclass(frozen=True)
class LoadedAgent:
    model: BaseModel
    processor: Any
    architecture: Architecture
    path: str


def load_agent(
    checkpoint: str | Path,
    device: torch.device,
    *,
    model: str | None = None,
    hidden_dim: int | None = None,
    features: str | None = None,
    frozen: bool = False,
) -> LoadedAgent:
    """Load ``checkpoint`` into the network it was trained as, in eval mode.

    The keyword arguments override what :func:`resolve_architecture` finds; leave
    them ``None`` to rebuild the checkpoint exactly as its run recorded it.
    ``frozen`` also turns gradients off, for a fixed opponent.
    """
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    architecture = resolve_architecture(
        checkpoint, state_dict, model=model, hidden_dim=hidden_dim, features=features
    )
    agent, processor = build_agent(architecture, device)
    try:
        agent.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(f"{checkpoint} does not fit {architecture}: {exc}") from exc
    agent.eval()
    if frozen:
        for param in agent.parameters():
            param.requires_grad_(False)
    return LoadedAgent(agent, processor, architecture, str(checkpoint))


__all__ = [
    "LoadedAgent",
    "build_agent",
    "infer_features",
    "infer_hidden_dim",
    "load_agent",
    "load_backend",
    "load_backend_spec",
    "resolve_architecture",
    "resolve_model_name",
    "run_record",
]
