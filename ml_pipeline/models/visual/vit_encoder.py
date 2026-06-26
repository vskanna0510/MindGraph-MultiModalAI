"""Vision Transformer visual encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.visual.base_visual_encoder import BaseVisualEncoder


class ViTEncoder(BaseVisualEncoder):
    backbone_name = "vit"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._vit = None
        self.pretrained = config.get("pretrained", "google/vit-base-patch16-224")

    def _load_vit(self) -> bool:
        if self._vit is not None:
            return True
        try:
            from transformers import ViTModel

            self._vit = ViTModel.from_pretrained(self.pretrained)
            return True
        except Exception:
            return False

    def _vit_features(self, x: torch.Tensor) -> torch.Tensor:
        if self._load_vit() and x.dim() == 4:
            out = self._vit(pixel_values=x).last_hidden_state
            return out
        return x

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any) -> EncoderOutput:
        x = self._vit_features(x)
        return super().encode(x, mask, **kwargs)
