"""Gated and dynamic fusion."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.output import FusionOutput


class GatedFusion(BaseFusionModule):
    strategy_name = "gated"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        names, scores, embs, quality_w = [], [], [], {}
        for name, emb in embeddings.items():
            pres = presence.get(name, torch.ones(emb.shape[0], device=emb.device))
            if pres.sum() == 0:
                continue
            score = self.gate(emb).squeeze(-1) * pres
            if quality_scores and name in quality_scores:
                score = score * quality_scores[name]
                quality_w[name] = quality_scores[name]
            names.append(name)
            scores.append(score)
            embs.append(emb)
        if not embs:
            b = next(iter(presence.values())).shape[0]
            z = torch.zeros(b, self.d_model, device=next(self.parameters()).device)
            return FusionOutput(fusion_embedding=z)
        stacked_scores = torch.stack(scores, dim=-1)
        w_all = F.softmax(stacked_scores, dim=-1)
        fused = sum(emb * w_all[:, i : i + 1] for i, emb in enumerate(embs))
        weights = {names[i]: w_all[:, i] for i in range(len(names))}
        return FusionOutput(
            fusion_embedding=fused,
            modality_weights=weights,
            quality_weights=quality_w,
            metadata={"strategy": "gated"},
        )


class DynamicFusion(GatedFusion):
    strategy_name = "dynamic"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        self.context = nn.GRU(d_model, d_model, batch_first=True)

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        out = super().fuse(embeddings, presence, quality_scores)
        seq = out.fusion_embedding.unsqueeze(1)
        dyn, _ = self.context(seq)
        out.fusion_embedding = dyn.squeeze(1)
        out.metadata["strategy"] = "dynamic"
        return out
