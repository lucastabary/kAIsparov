"""Typed contract every model backend must expose as ``BACKEND_SPEC``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kaisparov.models.base_model import BaseModel

CollectDataFn = Callable[..., None]
TrainOneEpochFn = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class BackendSpec:
    """Everything training needs from a model backend, checked once at load time."""

    name: str
    model_class: type[BaseModel]
    processor_class: type
    buffer_class: type
    collect_data: CollectDataFn
    train_one_epoch: TrainOneEpochFn
