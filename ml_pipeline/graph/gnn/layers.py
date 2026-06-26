"""Shared GNN utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_adj(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)
    if add_self_loops:
        eye = torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
        adj = adj + eye
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return adj / deg


def message_pass(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """adj: (B, N, N) or (N, N); x: (B, N, D) or (B, D) for single node."""
    if x.dim() == 2:
        x = x.unsqueeze(1)
    adj_n = normalize_adj(adj)
    if adj_n.dim() == 2:
        return torch.matmul(adj_n, x)
    return torch.bmm(adj_n, x)
