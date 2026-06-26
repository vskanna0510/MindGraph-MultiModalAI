"""Modality projection into shared latent space."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionProjection(nn.Module):
    """Linear → LayerNorm → GELU → Dropout → Residual → Output."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        h = self.dropout(torch.nn.functional.gelu(self.norm1(self.linear1(x))))
        return self.norm2(self.linear2(h)) + residual


class MultimodalProjection(nn.Module):
    def __init__(self, modalities: list[str], in_dim: int, latent_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({m: FusionProjection(in_dim, latent_dim, dropout) for m in modalities})

    def forward(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {m: self.proj[m](emb) for m, emb in embeddings.items() if m in self.proj}
