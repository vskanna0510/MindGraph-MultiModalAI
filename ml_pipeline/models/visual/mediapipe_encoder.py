"""MediaPipe landmark encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.visual.base_visual_encoder import BaseVisualEncoder


class MediaPipeEncoder(BaseVisualEncoder):
    backbone_name = "mediapipe"

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs: Any) -> EncoderOutput:
        landmarks = kwargs.get("landmarks")
        if landmarks is not None:
            x = landmarks
        return super().encode(x, mask, **kwargs)
