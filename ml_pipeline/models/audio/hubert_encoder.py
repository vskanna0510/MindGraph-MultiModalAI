"""HuBERT audio encoder."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.audio.wav2vec2_encoder import Wav2Vec2Encoder


class HuBERTEncoder(Wav2Vec2Encoder):
    backbone_name = "hubert"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = {**config, "pretrained": config.get("pretrained", "facebook/hubert-base-ls960")}
        super().__init__(cfg)
        self._hf_model = None

    def _load_hf(self) -> bool:
        if self._hf_model is not None:
            return True
        try:
            from transformers import HubertModel

            self._hf_model = HubertModel.from_pretrained(self.pretrained_name)
            return True
        except Exception:
            return False
