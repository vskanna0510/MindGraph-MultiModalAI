"""GraphSAGE layer."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.graph.gnn.layers import message_pass
from ml_pipeline.graph.gnn.base import BaseGraphModule


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        h_self = self.lin_self(x)
        if adj is not None:
            neigh = message_pass(adj, x)
            out = torch.relu(h_self + self.lin_neigh(neigh))
        else:
            out = torch.relu(h_self)
        out = self.dropout(out)
        return out.squeeze(1) if squeeze else out


class GraphSAGEModule(BaseGraphModule):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([GraphSAGELayer(dims[i], dims[i + 1], dropout) for i in range(num_layers)])
        self.out = nn.Linear(dims[-1], in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        h = x.unsqueeze(1) if adj is not None and x.dim() == 2 and adj.dim() == 3 else x
        for layer in self.layers:
            h = layer(h, adj)
        if h.dim() == 3:
            h = h.mean(dim=1)
        return self.out(h) + x
