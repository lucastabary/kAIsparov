from kaisparov.models.backend_spec import BackendSpec
from kaisparov.models.shared_rgcn.model import SharedRGCNModel
from kaisparov.models.shared_rgcn.processor import (
    PPOBuffer,
    SharedRGCNProcessor,
    compute_reward,
    get_legal_mask,
    train_one_epoch,
)
from kaisparov.training.rollout import collect_data

# Module-level hooks used by the rollout (looked up as attributes of this module).
PROCESSOR_CLASS = SharedRGCNProcessor

BACKEND_SPEC = BackendSpec(
    name=SharedRGCNModel.MODEL_NAME,
    model_class=SharedRGCNModel,
    processor_class=SharedRGCNProcessor,
    buffer_class=PPOBuffer,
    collect_data=collect_data,
    train_one_epoch=train_one_epoch,
)

__all__ = [
    "SharedRGCNModel",
    "SharedRGCNProcessor",
    "PROCESSOR_CLASS",
    "PPOBuffer",
    "collect_data",
    "train_one_epoch",
    "get_legal_mask",
    "compute_reward",
    "BACKEND_SPEC",
]
