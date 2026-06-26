"""Shared projection head: Linear → LayerNorm → GELU → Dropout → Projection → Normalize."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))
