"""Graph Convolutional Network."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.graph.gnn.layers import message_pass, normalize_adj
from ml_pipeline.graph.gnn.base import BaseGraphModule


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        if adj is None:
            return self.dropout(torch.relu(self.lin(x)))
        h = x.unsqueeze(1) if x.dim() == 2 else x
        out = torch.bmm(normalize_adj(adj), self.lin(h)) if adj.dim() == 3 else torch.matmul(normalize_adj(adj), self.lin(h))
        return self.dropout(torch.relu(out.squeeze(1) if out.dim() == 3 and out.size(1) == 1 else out))


class GCNModule(BaseGraphModule):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1], dropout) for i in range(num_layers)])
        self.out = nn.Linear(dims[-1], in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            if h.dim() == 3:
                h = h.mean(dim=1)
        return self.out(h) + x
