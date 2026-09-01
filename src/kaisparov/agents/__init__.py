"""Agents (policies).

Baselines (``RandomAgent``, ``MaterialAgent``) are dependency-light and import
eagerly. ``NeuralAgent`` pulls in torch, so it is imported lazily on first access
to keep baseline usage (and tests) fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaisparov.agents.base import Move, Policy
from kaisparov.agents.material_agent import MaterialAgent
from kaisparov.agents.random_agent import RandomAgent

if TYPE_CHECKING:
    from kaisparov.agents.neural_agent import NeuralAgent

__all__ = ["Policy", "Move", "RandomAgent", "MaterialAgent", "NeuralAgent"]


def __getattr__(name: str) -> Any:
    if name == "NeuralAgent":
        from kaisparov.agents.neural_agent import NeuralAgent

        return NeuralAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
