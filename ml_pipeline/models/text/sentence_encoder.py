"""Sentence transformer text encoder."""

from __future__ import annotations

from typing import Any

import torch

from ml_pipeline.models.text.base_text_encoder import BaseTextEncoder


class SentenceEncoder(BaseTextEncoder):
    backbone_name = "sentence_transformer"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pretrained = config.get("pretrained", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self._model = None

    def _encode_sentence(self, x: torch.Tensor) -> torch.Tensor:
        try:
            from sentence_transformers import SentenceTransformer

            if self._model is None:
                self._model = SentenceTransformer(self.pretrained)
            return torch.tensor(self._model.encode(["placeholder"] * x.shape[0])).unsqueeze(1)
        except Exception:
            return x.unsqueeze(1) if x.dim() == 2 else x

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any):
        if kwargs.get("use_sentence_model"):
            x = self._encode_sentence(x)
        return super().encode(x, mask, **kwargs)
