"""Audio encoder factory."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.audio.base_audio_encoder import BaseAudioEncoder
from ml_pipeline.models.audio.hubert_encoder import HuBERTEncoder
from ml_pipeline.models.audio.wav2vec2_encoder import Wav2Vec2Encoder
from ml_pipeline.models.audio.wavlm_encoder import WavLMEncoder
from ml_pipeline.models.base.encoder import ProjectionEncoder


AUDIO_ENCODERS = {
    "projection": None,
    "wav2vec2": Wav2Vec2Encoder,
    "wav2vec2_base": Wav2Vec2Encoder,
    "wav2vec2_large": Wav2Vec2Encoder,
    "hubert": HuBERTEncoder,
    "wavlm": WavLMEncoder,
}


def build_audio_encoder(config: dict[str, Any]):
    backbone = config.get("encoder", config.get("backbone", "wav2vec2"))
    cls = AUDIO_ENCODERS.get(backbone)
    if cls is None:
        return ProjectionEncoder(
            modality="audio",
            input_dim=int(config.get("input_dim", 768)),
            latent_dim=int(config.get("latent_dim", 512)),
            hidden_dim=int(config.get("hidden_dim", 512)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
        )
    return cls(config)


class AudioEncoder(Wav2Vec2Encoder):
    """Default audio encoder — Wav2Vec2 with feature fusion."""

    def __init__(self, config: dict) -> None:
        super().__init__(config)
