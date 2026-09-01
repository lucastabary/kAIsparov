"""Experiment tracking: write runs (RunManager) and query them (Registry).

``Registry`` is torch-free and imported eagerly; ``RunManager`` pulls in torch /
TensorBoard and is imported lazily so read-only tooling stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaisparov.tracking.registry import Registry

if TYPE_CHECKING:
    from kaisparov.tracking.run import RunManager

__all__ = ["Registry", "RunManager"]


def __getattr__(name: str) -> Any:
    if name == "RunManager":
        from kaisparov.tracking.run import RunManager

        return RunManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
