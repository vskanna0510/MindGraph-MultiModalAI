"""Training-only video augmentation."""

from __future__ import annotations

import numpy as np

from ml_pipeline.video.config import video_config


def maybe_augment_frame(frame: np.ndarray, split: str | None = None) -> np.ndarray:
    cfg = video_config().get("augmentation", {})
    if not cfg.get("enabled", False) or split not in (None, "train"):
        return frame
    out = frame.copy()
    rng = np.random.default_rng(42)
    if rng.random() < float(cfg.get("brightness_prob", 0.0)):
        out = np.clip(out.astype(np.float32) * rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
    if rng.random() < float(cfg.get("horizontal_flip_prob", 0.0)):
        out = np.fliplr(out)
    return out
