"""Dynamic GraphSAGE — time-conditioned aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.graph.gnn.graphsage import GraphSAGELayer
from ml_pipeline.graph.temporal.encoding import TemporalEncoder
from ml_pipeline.graph.gnn.base import BaseGraphModule


class DynamicGraphSAGEModule(BaseGraphModule):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal = TemporalEncoder(in_dim)
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([GraphSAGELayer(dims[i], dims[i + 1], dropout) for i in range(num_layers)])
        self.gate = nn.Sequential(nn.Linear(in_dim * 2, in_dim), nn.Sigmoid())
        self.out = nn.Linear(dims[-1], in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, time_delta: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            h = x
        else:
            t_emb = self.temporal(x, time_delta)
            gate = self.gate(torch.cat([x, t_emb], dim=-1))
            h = (x * (1 - gate) + t_emb * gate).unsqueeze(1)
        for layer in self.layers:
            h = layer(h, adj)
        if h.dim() == 3:
            h = h.mean(dim=1)
        base = x.mean(dim=1) if x.dim() == 3 else x
        return self.out(h) + base
