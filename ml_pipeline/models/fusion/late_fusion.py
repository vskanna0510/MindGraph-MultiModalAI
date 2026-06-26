"""Late fusion — ensemble modality-specific predictions."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.output import FusionOutput


class LateFusion(BaseFusionModule):
    strategy_name = "late"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        self.heads = nn.ModuleDict({m: nn.Linear(d_model, d_model) for m in ("audio", "visual", "text", "image")})
        self.combiner = nn.Linear(d_model, d_model)

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        reps, weights = [], {}
        for name, emb in embeddings.items():
            if name not in self.heads:
                continue
            pres = presence.get(name, torch.ones(emb.shape[0], device=emb.device))
            if pres.sum() == 0:
                continue
            rep = self.heads[name](emb) * pres.unsqueeze(-1)
            reps.append(rep)
            weights[name] = pres
        if not reps:
            b = next(iter(presence.values())).shape[0]
            z = torch.zeros(b, self.d_model, device=next(self.parameters()).device)
            return FusionOutput(fusion_embedding=z)
        stacked = torch.stack(reps, dim=0).mean(dim=0)
        fused = self.combiner(stacked)
        return FusionOutput(fusion_embedding=fused, modality_weights=weights, metadata={"strategy": "late"})
