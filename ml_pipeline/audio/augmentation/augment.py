"""Training-only audio augmentation."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config


def maybe_augment(audio: np.ndarray, sample_rate: int, split: str | None = None) -> np.ndarray:
    """Apply augmentation only for training split."""
    cfg = audio_config().get("augmentation", {})
    if not cfg.get("enabled", False) or split not in (None, "train"):
        return audio
    out = audio.copy()
    rng = np.random.default_rng(42)
    if rng.random() < float(cfg.get("noise_prob", 0.0)):
        out = out + rng.normal(0, 0.005, size=out.shape).astype(np.float32)
    if rng.random() < float(cfg.get("random_gain_prob", 0.0)):
        out = out * rng.uniform(0.9, 1.1)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
