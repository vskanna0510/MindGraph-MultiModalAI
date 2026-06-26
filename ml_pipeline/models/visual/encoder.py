"""Visual encoder factory."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.base.encoder import ProjectionEncoder
from ml_pipeline.models.visual.mediapipe_encoder import MediaPipeEncoder
from ml_pipeline.models.visual.videomae_encoder import VideoMAEEncoder
from ml_pipeline.models.visual.vit_encoder import ViTEncoder

VISUAL_ENCODERS = {
    "projection": None,
    "mediapipe": MediaPipeEncoder,
    "vit": ViTEncoder,
    "videomae": VideoMAEEncoder,
}


def build_visual_encoder(config: dict[str, Any]):
    backbone = config.get("encoder", config.get("backbone", "vit"))
    cls = VISUAL_ENCODERS.get(backbone)
    if cls is None:
        return ProjectionEncoder("visual", int(config.get("input_dim", 768)), int(config.get("latent_dim", 512)),
                                 int(config.get("hidden_dim", 512)), int(config.get("num_layers", 2)),
                                 float(config.get("dropout", 0.1)))
    return cls(config)


class VisualEncoder(ViTEncoder):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
