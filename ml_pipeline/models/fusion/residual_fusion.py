"""Residual fusion preserving modality pathways."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.output import FusionOutput
from ml_pipeline.models.fusion.transformer_fusion import TransformerFusion


class ResidualFusion(BaseFusionModule):
    strategy_name = "residual"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        self.inner = TransformerFusion(d_model, config)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        inner_out = self.inner.fuse(embeddings, presence, quality_scores)
        residual = torch.stack([e for n, e in embeddings.items() if presence.get(n, torch.ones(1)).sum() > 0], dim=0).mean(0)
        fused = inner_out.fusion_embedding + self.residual_scale * residual
        inner_out.fusion_embedding = fused
        inner_out.metadata["strategy"] = "residual"
        return inner_out
