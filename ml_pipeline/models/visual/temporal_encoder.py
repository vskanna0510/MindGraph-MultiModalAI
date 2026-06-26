"""Temporal visual modeling."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.models.encoders.pooling import AttentionPooling


class TemporalVisualEncoder(nn.Module):
    """Frame sequence → temporal self-attention → pooled visual embedding."""

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pooling = AttentionPooling(dim, "attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        pooled, attn = self.pooling(hidden, mask)
        return hidden, pooled, attn
