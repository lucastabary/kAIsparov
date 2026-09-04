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
    from kaisparov.agents.minimax_agent import MinimaxAgent
    from kaisparov.agents.neural_agent import NeuralAgent
    from kaisparov.agents.neural_analyzer import NeuralAnalyzer

__all__ = [
    "Policy",
    "Move",
    "RandomAgent",
    "MaterialAgent",
    "NeuralAgent",
    "MinimaxAgent",
    "NeuralAnalyzer",
]

_LAZY = {
    "NeuralAgent": "kaisparov.agents.neural_agent",
    "MinimaxAgent": "kaisparov.agents.minimax_agent",
    "NeuralAnalyzer": "kaisparov.agents.neural_analyzer",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
