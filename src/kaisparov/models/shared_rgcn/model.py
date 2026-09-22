import torch
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, RGCNConv

from kaisparov.models.base_model import BaseModel
from kaisparov.models.features import DEFAULT_FEATURES, get_feature_set


class SharedChessRGCN(torch.nn.Module):
    """Weight-tied R-GCN backbone: one relational conv, applied ``num_steps`` times.

    Same relational bias as ``rgcn`` (one MLP per relation), but a single set of
    per-relation weights is reused at every message-passing step instead of one set
    per step. A linear encoder lifts the input features to ``hidden_channels`` so the
    shared conv is always a ``hidden -> hidden`` map, and a linear decoder produces
    the node embeddings; depth then costs no parameters at all.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_steps=4):
        super().__init__()
        self.num_steps = int(num_steps)
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}.")

        self.encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.conv = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.decoder = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        x = torch.relu(self.encoder(x))

        # The same conv at every step, residually — message passing becomes an
        # iterated operator rather than a stack of distinct layers.
        for _ in range(self.num_steps):
            h = self.conv(x, edge_index, edge_type)
            x = torch.relu(x + h)

        return self.decoder(x)


class SharedRGCNModel(BaseModel):
    """``rgcn``'s actor-critic with its message-passing steps tied to one weight set."""

    MODEL_NAME = "shared_rgcn"
    NUM_STEPS = 4  # message-passing steps; depth is free (weights are shared)

    def __init__(self, hidden_dim=8, num_steps=NUM_STEPS, features=DEFAULT_FEATURES):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        # The node feature set this model reads (see RGCNModel.features).
        self.features = features
        self.chess_rgcn = SharedChessRGCN(
            in_channels=get_feature_set(features).dim,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_relations=6,  # 6 edge types
            num_steps=num_steps,
        )

        self.aggregation = AttentionalAggregation(
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim, 1),
            )
        )
        self.actor_head = torch.nn.Sequential(
            torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),  # Output a score for each node
        )

        self.critic_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),  # Output a value for the state
        )

    def forward(self, data: Data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        x = self.chess_rgcn(x, edge_index, edge_type)

        h_src, h_des = x[edge_index[0]], x[edge_index[1]]
        edge_rep = torch.cat([h_src, h_des], dim=-1)
        action_scores = self.actor_head(edge_rep).squeeze(-1)  # [E]

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        board_state = self.aggregation(x, index=batch)
        state_value = self.critic_head(board_state).squeeze(-1)  # [B]

        return action_scores, state_value
