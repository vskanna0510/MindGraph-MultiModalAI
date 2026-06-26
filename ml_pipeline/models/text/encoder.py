"""Text encoder factory."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.base.encoder import ProjectionEncoder
from ml_pipeline.models.text.indicbert_encoder import IndicBERTEncoder
from ml_pipeline.models.text.mbert_encoder import MBERTEncoder
from ml_pipeline.models.text.sentence_encoder import SentenceEncoder
from ml_pipeline.models.text.xlmr_encoder import XLMRoBERTaEncoder

TEXT_ENCODERS = {
    "projection": None,
    "indicbert": IndicBERTEncoder,
    "mbert": MBERTEncoder,
    "xlmr": XLMRoBERTaEncoder,
    "sentence_transformer": SentenceEncoder,
}


def build_text_encoder(config: dict[str, Any]):
    backbone = config.get("encoder", config.get("backbone", "indicbert"))
    cls = TEXT_ENCODERS.get(backbone)
    if cls is None:
        return ProjectionEncoder("text", int(config.get("input_dim", 768)), int(config.get("latent_dim", 512)),
                                int(config.get("hidden_dim", 512)), int(config.get("num_layers", 2)),
                                float(config.get("dropout", 0.1)))
    return cls(config)


class TextEncoder(IndicBERTEncoder):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
