"""Audio encoder projection layers."""

from __future__ import annotations

import torch.nn as nn

from ml_pipeline.models.encoders.projection_head import ProjectionHead


class AudioInputProjection(nn.Module):
    """Project raw acoustic features to backbone dimension."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = ProjectionHead(in_dim, out_dim, out_dim, dropout)

    def forward(self, x):
        return self.proj(x)
