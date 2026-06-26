"""Early fusion — concatenate projected embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.output import FusionOutput


class EarlyFusion(BaseFusionModule):
    strategy_name = "early"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        self.merger = nn.Linear(d_model * 4, d_model)

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        parts, weights = [], {}
        for name, emb in embeddings.items():
            pres = presence.get(name)
            if pres is not None and pres.sum() == 0:
                continue
            parts.append(emb)
            weights[name] = pres if pres is not None else torch.ones(emb.shape[0], device=emb.device)
        if not parts:
            b = next(iter(presence.values())).shape[0]
            z = torch.zeros(b, self.d_model, device=next(self.parameters()).device)
            return FusionOutput(fusion_embedding=z)
        while len(parts) < 4:
            parts.append(torch.zeros_like(parts[0]))
        cat = torch.cat(parts[:4], dim=-1)
        fused = self.merger(cat)
        return FusionOutput(fusion_embedding=fused, modality_weights=weights, metadata={"strategy": "early"})
