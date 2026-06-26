"""Base text encoder with multilingual support."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.encoders.base import BaseModalityEncoder
from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.encoders.pooling import AttentionPooling
from ml_pipeline.models.text.projection import TextInputProjection


class BaseTextEncoder(BaseModalityEncoder):
    modality = "text"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        hidden = int(config.get("hidden_dim", 512))
        self.input_proj = TextInputProjection(int(config.get("input_dim", 768)), hidden, self.dropout)
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, int(config.get("num_heads", 8)), hidden * 4, self.dropout, batch_first=True),
            num_layers=int(config.get("num_layers", 2)),
        )
        self.pooling = AttentionPooling(hidden, config.get("pooling", "attention"))
        self.refinement = self.build_refinement(hidden)
        self.output_head = nn.Linear(hidden, self.latent_dim)
        self.confidence_head = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.Sigmoid())
        self.lang_embed = nn.Embedding(16, hidden)

    def encode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> EncoderOutput:
        hidden = self.input_proj(x)
        if language_ids is not None:
            hidden = hidden + self.lang_embed(language_ids.clamp(0, 15)).unsqueeze(1)
        hidden = self.backbone(hidden)
        pooled, attn = self.pooling(hidden, mask)
        refined = self.refinement(pooled)
        embedding = nn.functional.normalize(self.output_head(refined), dim=-1)
        return EncoderOutput(
            embedding=embedding,
            hidden_states=hidden,
            attention_maps=attn,
            intermediate={"token_mean": hidden.mean(dim=1), "token_max": hidden.max(dim=1).values},
            confidence=self.confidence_head(embedding).squeeze(-1),
            metadata={"backbone": self.backbone_name, "languages": ["en", "ta", "hi", "mixed"]},
        )
