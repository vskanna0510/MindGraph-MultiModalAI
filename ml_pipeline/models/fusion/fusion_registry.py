"""Fusion strategy registry and factory."""

from __future__ import annotations

from typing import Any, Type

import torch.nn as nn

from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.co_attention import CoAttentionFusion
from ml_pipeline.models.fusion.cross_attention import CrossAttentionFusion
from ml_pipeline.models.fusion.early_fusion import EarlyFusion
from ml_pipeline.models.fusion.gated_fusion import DynamicFusion, GatedFusion
from ml_pipeline.models.fusion.late_fusion import LateFusion
from ml_pipeline.models.fusion.quality_aware import QualityAwareFusion
from ml_pipeline.models.fusion.residual_fusion import ResidualFusion
from ml_pipeline.models.fusion.transformer_fusion import TransformerFusion

FUSION_REGISTRY: dict[str, Type[BaseFusionModule]] = {
    "early": EarlyFusion,
    "late": LateFusion,
    "cross_attention": CrossAttentionFusion,
    "co_attention": CoAttentionFusion,
    "transformer": TransformerFusion,
    "cross_modal_transformer": TransformerFusion,
    "gated": GatedFusion,
    "dynamic": DynamicFusion,
    "residual": ResidualFusion,
    "quality_aware": QualityAwareFusion,
}


def list_fusion_strategies() -> list[str]:
    return sorted(FUSION_REGISTRY.keys())


def build_fusion(strategy: str, d_model: int, config: dict[str, Any] | None = None) -> BaseFusionModule:
    key = strategy.lower().replace("-", "_")
    if key not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion strategy '{strategy}'. Available: {list_fusion_strategies()}")
    return FUSION_REGISTRY[key](d_model, config)


class MultimodalFusionStack(nn.Module):
    """Projection → fusion strategy → optional temporal fusion."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        fcfg = cfg.get("fusion", cfg)
        self.d_model = int(fcfg.get("d_model", 512))
        dropout = float(fcfg.get("dropout", 0.1))
        modalities = list(fcfg.get("modalities", ["audio", "visual", "text", "image"]))
        strategy = str(fcfg.get("strategy", "cross_modal_transformer"))

        from ml_pipeline.models.fusion.projection import MultimodalProjection

        self.projection = MultimodalProjection(modalities, self.d_model, self.d_model, dropout)
        self.fusion = build_fusion(strategy, self.d_model, fcfg)
        tcfg = cfg.get("temporal", {})
        self.temporal_enabled = bool(tcfg.get("enabled", True))
        if self.temporal_enabled:
            from ml_pipeline.models.fusion.temporal_fusion import TemporalFusion

            self.temporal = TemporalFusion(self.d_model, {**fcfg, **tcfg})
        else:
            self.temporal = None

    def forward(self, embeddings, presence, quality_scores=None, history=None, session_idx=None):
        from ml_pipeline.models.fusion.output import FusionOutput

        projected = self.projection(embeddings)
        out = self.fusion.fuse(projected, presence, quality_scores)
        if self.temporal is not None:
            temporal_out = self.temporal(out.fusion_embedding, history, session_idx)
            out.temporal_embedding = temporal_out.temporal_embedding
            out.attention_maps.update(temporal_out.attention_maps)
            out.historical_weights = temporal_out.historical_weights
            out.hidden_states = temporal_out.hidden_states
        return out

    def fused_tensor(self, out) -> "torch.Tensor":
        import torch

        if out.temporal_embedding is not None:
            return out.temporal_embedding
        return out.fusion_embedding
