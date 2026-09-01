from kaisparov.models.backend_spec import BackendSpec
from kaisparov.models.gnn_v1.model import GNN1Model
from kaisparov.models.gnn_v1.processor import (
    GNN1Processor,
    PPOBuffer,
    compute_reward,
    get_legal_mask,
    train_one_epoch,
)
from kaisparov.training.rollout import collect_data

# Module-level hooks used by the rollout (looked up as attributes of this module).
PROCESSOR_CLASS = GNN1Processor

BACKEND_SPEC = BackendSpec(
    name=GNN1Model.MODEL_NAME,
    model_class=GNN1Model,
    processor_class=GNN1Processor,
    buffer_class=PPOBuffer,
    collect_data=collect_data,
    train_one_epoch=train_one_epoch,
)

__all__ = [
    "GNN1Model",
    "GNN1Processor",
    "PROCESSOR_CLASS",
    "PPOBuffer",
    "collect_data",
    "train_one_epoch",
    "get_legal_mask",
    "compute_reward",
    "BACKEND_SPEC",
]
