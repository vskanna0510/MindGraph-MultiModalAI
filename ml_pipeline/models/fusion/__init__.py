"""Multimodal fusion package."""

from ml_pipeline.models.fusion.base_fusion import BaseFusionModule
from ml_pipeline.models.fusion.fusion_registry import MultimodalFusionStack, build_fusion, list_fusion_strategies
from ml_pipeline.models.fusion.output import FusionOutput
from ml_pipeline.models.fusion.projection import FusionProjection, MultimodalProjection

__all__ = [
    "BaseFusionModule",
    "FusionOutput",
    "FusionProjection",
    "MultimodalProjection",
    "MultimodalFusionStack",
    "build_fusion",
    "list_fusion_strategies",
]
