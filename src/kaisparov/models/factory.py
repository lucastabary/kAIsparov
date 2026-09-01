"""Dynamic backend loading by name."""

from __future__ import annotations

import importlib

from kaisparov.models.backend_spec import BackendSpec

DEFAULT_MODEL = "gnn_v1"
MODEL_MODULES = {
    "gnn_v1": "kaisparov.models.gnn_v1",
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
