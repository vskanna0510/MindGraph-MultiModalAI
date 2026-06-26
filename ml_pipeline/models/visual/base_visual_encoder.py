"""Base visual encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.encoders.base import BaseModalityEncoder
from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.encoders.pooling import AttentionPooling
from ml_pipeline.models.visual.projection import VisualInputProjection
from ml_pipeline.models.visual.temporal_encoder import TemporalVisualEncoder


class BaseVisualEncoder(BaseModalityEncoder):
    modality = "visual"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        in_dim = int(config.get("input_dim", 768))
        hidden = int(config.get("hidden_dim", 512))
        self.input_proj = VisualInputProjection(in_dim, hidden, self.dropout)
        self.temporal = TemporalVisualEncoder(
            hidden,
            int(config.get("temporal_layers", 2)),
            int(config.get("num_heads", 8)),
            self.dropout,
        )
        self.refinement = self.build_refinement(hidden)
        self.output_head = nn.Linear(hidden, self.latent_dim)
        self.confidence_head = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.Sigmoid())

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any) -> EncoderOutput:
        hidden = self.input_proj(x)
        temporal_hidden, pooled, attn = self.temporal(hidden, mask)
        refined = self.refinement(pooled)
        embedding = nn.functional.normalize(self.output_head(refined), dim=-1)
        return EncoderOutput(
            embedding=embedding,
            hidden_states=temporal_hidden,
            attention_maps=attn,
            intermediate={"temporal": pooled},
            confidence=self.confidence_head(embedding).squeeze(-1),
            metadata={"backbone": self.backbone_name},
        )
