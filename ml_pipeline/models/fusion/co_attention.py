"""Co-attention fusion."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.cross_attention import CrossAttentionFusion
from ml_pipeline.models.fusion.output import FusionOutput


class CoAttentionFusion(CrossAttentionFusion):
    strategy_name = "co_attention"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        self.co_layers = nn.ModuleList([nn.MultiheadAttention(d_model, 8, batch_first=True) for _ in range(2)])

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        names = [n for n, e in embeddings.items() if presence.get(n, torch.ones(1)).sum() > 0]
        if len(names) < 2:
            return super().fuse(embeddings, presence, quality_scores)
        a, b = names[0], names[1]
        qa = embeddings[a].unsqueeze(1)
        qb = embeddings[b].unsqueeze(1)
        for layer in self.co_layers:
            qa, _ = layer(qa, qb, qb)
            qb, _ = layer(qb, qa, qa)
        fused = (qa.squeeze(1) + qb.squeeze(1)) / 2
        return FusionOutput(
            fusion_embedding=fused,
            modality_weights={a: presence[a], b: presence[b]},
            metadata={"strategy": "co_attention", "pairs": [a, b]},
        )
