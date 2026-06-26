"""Quality-aware fusion weighting."""

from __future__ import annotations

from typing import Any

import torch

from ml_pipeline.models.fusion.gated_fusion import GatedFusion
from ml_pipeline.models.fusion.output import FusionOutput


class QualityAwareFusion(GatedFusion):
    strategy_name = "quality_aware"

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        if quality_scores is None:
            quality_scores = {n: torch.ones(e.shape[0], device=e.device) for n, e in embeddings.items()}
        out = super().fuse(embeddings, presence, quality_scores)
        out.metadata["strategy"] = "quality_aware"
        total_q = torch.stack(list(out.quality_weights.values())).mean(dim=0) if out.quality_weights else None
        out.confidence = total_q
        return out
