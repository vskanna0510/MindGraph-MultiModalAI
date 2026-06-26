"""mBERT text encoder."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.text.indicbert_encoder import IndicBERTEncoder


class MBERTEncoder(IndicBERTEncoder):
    backbone_name = "mbert"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__({**config, "pretrained": config.get("pretrained", "bert-base-multilingual-cased")})
