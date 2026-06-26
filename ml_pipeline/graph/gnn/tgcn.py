"""Temporal Graph Convolutional Network."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.graph.gnn.gcn import GCNLayer
from ml_pipeline.graph.temporal.encoding import TemporalEncoder
from ml_pipeline.graph.gnn.base import BaseGraphModule


class TGCNModule(BaseGraphModule):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.temporal = TemporalEncoder(in_dim)
        self.layers = nn.ModuleList([GCNLayer(in_dim if i == 0 else hidden_dim, hidden_dim, dropout) for i in range(num_layers)])
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None, time_delta: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 3:
            h = x
        else:
            h = self.temporal(x, time_delta).unsqueeze(1)
        for layer in self.layers:
            h = layer(h, adj)
            if h.dim() == 3 and h.size(1) == 1:
                h = h.squeeze(1)
        if h.dim() == 3:
            h = h.mean(dim=1)
        base = x.mean(dim=1) if x.dim() == 3 else x
        return self.out(h) + base
