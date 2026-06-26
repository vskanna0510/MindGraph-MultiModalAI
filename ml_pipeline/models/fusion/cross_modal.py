"""Cross-modal and temporal fusion (backward-compatible wrappers)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.models.base.fusion import BaseFusion
from ml_pipeline.models.fusion.transformer_fusion import TransformerFusion


class CrossModalTransformer(BaseFusion):
    """Legacy API — delegates to TransformerFusion."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self._inner = TransformerFusion(
            d_model,
            {"num_heads": num_heads, "num_layers": num_layers, "dropout": dropout},
        )

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        presence: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        out = self._inner.fuse(embeddings, presence)
        return out.fusion_embedding


class TemporalFusion(nn.Module):
    """Legacy API — delegates to temporal fusion module."""

    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        from ml_pipeline.models.fusion.temporal_fusion import TemporalFusion as TemporalFusionCore

        self._inner = TemporalFusionCore(d_model, {"num_layers": num_layers, "dropout": dropout})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._inner(x, history=None)
        return out.temporal_embedding if out.temporal_embedding is not None else x
