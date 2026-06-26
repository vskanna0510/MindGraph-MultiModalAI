"""Image encoder factory."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.base.encoder import ProjectionEncoder
from ml_pipeline.models.image.clip_encoder import CLIPEncoder
from ml_pipeline.models.image.vit_image_encoder import ViTImageEncoder

IMAGE_ENCODERS = {
    "projection": None,
    "vit": ViTImageEncoder,
    "clip": CLIPEncoder,
}


def build_image_encoder(config: dict[str, Any]):
    backbone = config.get("encoder", config.get("backbone", "vit"))
    cls = IMAGE_ENCODERS.get(backbone)
    if cls is None:
        return ProjectionEncoder("image", int(config.get("input_dim", 512)), int(config.get("latent_dim", 512)),
                                int(config.get("hidden_dim", 512)), int(config.get("num_layers", 2)),
                                float(config.get("dropout", 0.1)))
    return cls(config)


class ImageEncoder(ViTImageEncoder):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
