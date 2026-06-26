"""Cross-attention fusion between modality pairs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.fusion.attention_pooling import FusionAttentionPooling
from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.output import FusionOutput


class CrossAttentionFusion(BaseFusionModule):
    strategy_name = "cross_attention"

    def __init__(self, d_model: int, config: dict[str, Any] | None = None) -> None:
        super().__init__(d_model, config)
        self.cross = nn.MultiheadAttention(d_model, int(config.get("num_heads", 8) if config else 8), batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, int(config.get("num_heads", 8) if config else 8), batch_first=True)
        self.pool = FusionAttentionPooling(d_model)
        self.mode = (config or {}).get("cross_attention_mode", "all_to_all")

    def fuse(self, embeddings, presence, quality_scores=None) -> FusionOutput:
        tokens, names, attn_maps = [], [], {}
        for name, emb in embeddings.items():
            pres = presence.get(name)
            if pres is not None and pres.sum() == 0:
                continue
            seq = emb.unsqueeze(1)
            if (self.config or {}).get("self_attention_before_cross", True):
                seq, _ = self.self_attn(seq, seq, seq)
                attn_maps[f"self_{name}"] = torch.ones(seq.shape[0], 1, device=seq.device)
            tokens.append(seq)
            names.append(name)
        if not tokens:
            b = next(iter(presence.values())).shape[0]
            z = torch.zeros(b, self.d_model, device=next(self.parameters()).device)
            return FusionOutput(fusion_embedding=z)
        stacked = torch.cat(tokens, dim=1)
        cross_out, cross_attn = self.cross(stacked, stacked, stacked)
        pooled, pool_attn = self.pool(cross_out)
        attn_maps["cross"] = cross_attn.mean(dim=1) if cross_attn is not None else pool_attn
        weights = {n: presence.get(n, torch.ones(pooled.shape[0], device=pooled.device)) for n in names}
        return FusionOutput(
            fusion_embedding=pooled,
            attention_maps=attn_maps,
            modality_weights=weights,
            hidden_states=cross_out,
            metadata={"strategy": "cross_attention", "mode": self.mode},
        )
