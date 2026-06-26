"""VideoMAE visual encoder."""

from __future__ import annotations

from typing import Any

from ml_pipeline.models.visual.vit_encoder import ViTEncoder


class VideoMAEEncoder(ViTEncoder):
    backbone_name = "videomae"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = {**config, "pretrained": config.get("pretrained", "MCG-NJU/videomae-base")}
        super().__init__(cfg)
        self._vit = None

    def _load_vit(self) -> bool:
        if self._vit is not None:
            return True
        try:
            from transformers import VideoMAEModel

            self._vit = VideoMAEModel.from_pretrained(self.pretrained)
            return True
        except Exception:
            return False
