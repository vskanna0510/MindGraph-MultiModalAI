"""Base audio encoder pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.audio.attention_pooling import AudioAttentionPooling
from ml_pipeline.models.audio.projection import AudioInputProjection
from ml_pipeline.models.encoders.base import BaseModalityEncoder
from ml_pipeline.models.encoders.output import EncoderOutput


class BaseAudioEncoder(BaseModalityEncoder):
    modality = "audio"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        in_dim = int(config.get("input_dim", 768))
        hidden = int(config.get("hidden_dim", 512))
        self.input_proj = AudioInputProjection(in_dim, hidden, self.dropout)
        self.backbone = self._build_backbone(config, hidden)
        self.pooling = AudioAttentionPooling(hidden, self.pooling_mode)
        self.refinement = self.build_refinement(hidden)
        self.output_head = nn.Linear(hidden, self.latent_dim)
        self.confidence_head = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.Sigmoid())

    def _build_backbone(self, config: dict, hidden: int) -> nn.Module:
        layers = int(config.get("backbone_layers", 2))
        mods: list[nn.Module] = []
        for _ in range(layers):
            mods.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(self.dropout)])
        return nn.Sequential(*mods)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any) -> EncoderOutput:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            if x.shape[-1] < self.config.get("input_dim", 768):
                x = nn.functional.pad(x, (0, int(self.config["input_dim"]) - x.shape[-1]))
            x = x.unsqueeze(1) if x.dim() == 2 else x
        hidden = self.input_proj(x)
        hidden = self.backbone(hidden)
        pooled, attn = self.pooling(hidden, mask)
        refined = self.refinement(pooled)
        embedding = self.output_head(refined)
        embedding = nn.functional.normalize(embedding, dim=-1)
        confidence = self.confidence_head(embedding).squeeze(-1)
        return EncoderOutput(
            embedding=embedding,
            hidden_states=hidden,
            attention_maps=attn,
            intermediate={"pooled": pooled, "refined": refined},
            confidence=confidence,
            metadata={"backbone": self.backbone_name, "pooling": self.pooling_mode},
        )
