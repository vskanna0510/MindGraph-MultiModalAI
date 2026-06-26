"""XLM-RoBERTa text encoder."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.text.indicbert_encoder import IndicBERTEncoder


class XLMRoBERTaEncoder(IndicBERTEncoder):
    backbone_name = "xlmr"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__({**config, "pretrained": config.get("pretrained", "xlm-roberta-base")})
