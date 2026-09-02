from kaisparov.models.backend_spec import BackendSpec
from kaisparov.models.rgcn.model import RGCNModel
from kaisparov.models.rgcn.processor import (
    PPOBuffer,
    RGCNProcessor,
    compute_reward,
    get_legal_mask,
    train_one_epoch,
)
from kaisparov.training.rollout import collect_data

# Module-level hooks used by the rollout (looked up as attributes of this module).
PROCESSOR_CLASS = RGCNProcessor

BACKEND_SPEC = BackendSpec(
    name=RGCNModel.MODEL_NAME,
    model_class=RGCNModel,
    processor_class=RGCNProcessor,
    buffer_class=PPOBuffer,
    collect_data=collect_data,
    train_one_epoch=train_one_epoch,
)

__all__ = [
    "RGCNModel",
    "RGCNProcessor",
    "PROCESSOR_CLASS",
    "PPOBuffer",
    "collect_data",
    "train_one_epoch",
    "get_legal_mask",
    "compute_reward",
    "BACKEND_SPEC",
]
