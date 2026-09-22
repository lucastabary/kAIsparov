"""Minimal base class for trainable model backends.

Persistence and experiment metadata are handled by :mod:`kaisparov.tracking`
(the ``runs/`` registry), and loading a checkpoint by
:func:`kaisparov.models.factory.load_agent`, so this base only covers construction.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

import torch


class BaseModel(torch.nn.Module, ABC):
    MODEL_NAME = "base-model"

    @classmethod
    def create_agent(cls, device: torch.device, **kwargs: Any) -> BaseModel:
        """Construct the model and move it to ``device``."""
        return cls(**kwargs).to(device)

    @staticmethod
    def create_optimizer(
        agent: torch.nn.Module, learning_rate: float = 1e-3
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(agent.parameters(), lr=learning_rate)
