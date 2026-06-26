"""Graph Attention Network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.graph.gnn.base import BaseGraphModule


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.randn(heads, 2 * out_dim) * 0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, n, _ = x.shape
        h = self.lin(x).view(b, n, self.heads, self.out_dim)
        if n == 1:
            return self.dropout(torch.relu(h.mean(dim=2))).squeeze(1)
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1, -1)
        e = F.leaky_relu((torch.cat([h_i, h_j], dim=-1) * self.attn).sum(dim=-1), 0.2)
        if adj is not None and adj.dim() == 3:
            e = e.masked_fill(adj.unsqueeze(-1).expand(-1, -1, -1, self.heads) == 0, float("-inf"))
        alpha = self.dropout(F.softmax(e, dim=2))
        out = torch.einsum("bijn,bjnh->binh", alpha, h)
        pooled = out.mean(dim=1)
        return pooled.squeeze(1) if pooled.size(1) == 1 else pooled


class GATModule(BaseGraphModule):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1, heads: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GATLayer(in_dim if i == 0 else hidden_dim, hidden_dim, heads, dropout) for i in range(num_layers)])
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adj)
            if h.dim() == 3:
                h = h.mean(dim=1)
        if h.dim() == 3:
            h = h.mean(dim=1)
        return self.out(h) + (x.mean(dim=1) if x.dim() == 3 else x)
