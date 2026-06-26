"""CLIP image encoder."""

from __future__ import annotations

from typing import Any

import torch

from ml_pipeline.models.image.vit_image_encoder import BaseImageEncoder


class CLIPEncoder(BaseImageEncoder):
    backbone_name = "clip"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pretrained = config.get("pretrained", "openai/clip-vit-base-patch32")
        self._model = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import CLIPVisionModel

            self._model = CLIPVisionModel.from_pretrained(self.pretrained)
            return True
        except Exception:
            return False

    def encode(self, x, mask=None, **kwargs):
        if x.dim() == 4 and self._load():
            x = self._model(pixel_values=x).last_hidden_state
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        return super().encode(x, mask, **kwargs)
