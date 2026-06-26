"""Stage 5 — Channel normalization."""

from __future__ import annotations

import numpy as np


def to_mono(audio: np.ndarray, original_channels: int = 1) -> np.ndarray:
    """Convert stereo/multi-channel to mono via weighted average."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=-1).astype(np.float32)
