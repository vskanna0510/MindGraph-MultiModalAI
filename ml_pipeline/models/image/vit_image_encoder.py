"""ViT image encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_pipeline.models.encoders.base import BaseModalityEncoder
from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.image.projection import ImageInputProjection


class BaseImageEncoder(BaseModalityEncoder):
    modality = "image"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        hidden = int(config.get("hidden_dim", 512))
        self.input_proj = ImageInputProjection(int(config.get("input_dim", 512)), hidden, self.dropout)
        self.refinement = self.build_refinement(hidden)
        self.output_head = nn.Linear(hidden, self.latent_dim)
        self.confidence_head = nn.Sequential(nn.Linear(self.latent_dim, 1), nn.Sigmoid())

    def encode(self, x, mask=None, **kwargs) -> EncoderOutput:
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        hidden = self.input_proj(x)
        pooled = hidden.mean(dim=1)
        refined = self.refinement(pooled)
        embedding = F.normalize(self.output_head(refined), dim=-1)
        return EncoderOutput(
            embedding=embedding,
            hidden_states=hidden,
            attention_maps=None,
            confidence=self.confidence_head(embedding).squeeze(-1),
            metadata={"backbone": self.backbone_name},
        )


class ViTImageEncoder(BaseImageEncoder):
    backbone_name = "vit"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pretrained = config.get("pretrained", "google/vit-base-patch16-224")
        self._vit = None

    def _load(self) -> bool:
        if self._vit is not None:
            return True
        try:
            from transformers import ViTModel

            self._vit = ViTModel.from_pretrained(self.pretrained)
            return True
        except Exception:
            return False

    def encode(self, x, mask=None, **kwargs) -> EncoderOutput:
        if x.dim() == 4 and self._load():
            import torch

            x = self._vit(pixel_values=x).last_hidden_state
        return super().encode(x, mask, **kwargs)
