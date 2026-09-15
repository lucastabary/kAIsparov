"""Dynamic backend loading by name."""

from __future__ import annotations

import importlib

from kaisparov.models.backend_spec import BackendSpec

DEFAULT_MODEL = "rgcn"
MODEL_MODULES = {
    "rgcn": "kaisparov.models.rgcn",
    "shared_rgcn": "kaisparov.models.shared_rgcn",
}


def resolve_model_name(model_name: str | None = None) -> str:
    return model_name or DEFAULT_MODEL


def load_backend(model_name: str | None = None):
    resolved = resolve_model_name(model_name)
    return importlib.import_module(MODEL_MODULES.get(resolved, resolved))


# state_dict keys whose first dimension equals hidden_dim, tried in order.
_HIDDEN_DIM_KEYS = ("chess_rgcn.conv1.bias", "critic_head.0.bias", "actor_head.0.bias")


def infer_hidden_dim(state_dict) -> int | None:
    """Read hidden_dim off a checkpoint so it need not be passed on the CLI.

    Checkpoints are plain ``state_dict``s with no metadata, but every candidate key
    is a 1-D tensor of length ``hidden_dim`` — so its shape tells us the width the
    model was trained at, tracked run or raw path alike.
    """
    for key in _HIDDEN_DIM_KEYS:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.dim() >= 1:
            return int(tensor.shape[0])
    return None


def load_backend_spec(model_name: str | None = None) -> BackendSpec:
    backend = load_backend(model_name)
    spec = getattr(backend, "BACKEND_SPEC", None)
    if spec is None:
        raise AttributeError(f"Model backend '{backend.__name__}' must expose BACKEND_SPEC.")
    if not isinstance(spec, BackendSpec):
        raise TypeError(f"BACKEND_SPEC in '{backend.__name__}' must be a BackendSpec.")
    return spec
