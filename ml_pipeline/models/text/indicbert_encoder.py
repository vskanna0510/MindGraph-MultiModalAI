"""IndicBERT text encoder."""

from __future__ import annotations

from typing import Any

import torch

from ml_pipeline.models.text.base_text_encoder import BaseTextEncoder


class IndicBERTEncoder(BaseTextEncoder):
    backbone_name = "indicbert"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pretrained = config.get("pretrained", "ai4bharat/indic-bert")
        self._model = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(self.pretrained)
            return True
        except Exception:
            return False

    def _embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if self._load():
            out = self._model(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state
        return torch.randn(input_ids.shape[0], input_ids.shape[1], int(self.config.get("hidden_dim", 512)))

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any):
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            x = self._embed(input_ids, mask)
        return super().encode(x, mask, **kwargs)
