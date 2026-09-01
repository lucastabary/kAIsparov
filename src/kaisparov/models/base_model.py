"""Minimal base class for trainable model backends.

Persistence and experiment metadata are handled by :mod:`kaisparov.tracking`
(the ``runs/`` registry), so this base only covers construction and loading a
plain ``state_dict`` for inference.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
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

    @classmethod
    def load_agent_for_inference(
        cls,
        device: torch.device,
        model_path: str | Path,
        **kwargs: Any,
    ) -> tuple[BaseModel, str]:
        """Build the model and load a checkpoint (a plain ``state_dict``) into it."""
        agent = cls.create_agent(device=device, **kwargs)
        state_dict = torch.load(model_path, map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent, str(model_path)
