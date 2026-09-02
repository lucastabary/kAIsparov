import torch
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, RGCNConv

from kaisparov.models.base_model import BaseModel


class ChessRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv4 = RGCNConv(hidden_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = torch.relu(self.conv1(x, edge_index, edge_type))

        h = self.conv2(x, edge_index, edge_type)
        x = torch.relu(x + h)

        h = self.conv3(x, edge_index, edge_type)
        x = torch.relu(x + h)

        x = self.conv4(x, edge_index, edge_type)

        return x


class RGCNModel(BaseModel):
    """GNN actor-critic model for chess graph states."""

    INPUT_DIM = 12  # 6 for ally pieces + 6 for enemy pieces
    MODEL_NAME = "rgcn"

    def __init__(self, hidden_dim=8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.chess_rgcn = ChessRGCN(
            in_channels=self.INPUT_DIM,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_relations=6,
        )  # 6 edge types

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
