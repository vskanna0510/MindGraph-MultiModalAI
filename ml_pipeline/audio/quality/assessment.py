"""Stage 3 — Audio quality assessment."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config
from ml_pipeline.audio.types import AudioMetadata


def assess_quality(audio: np.ndarray, metadata: AudioMetadata) -> tuple[float, list[str]]:
    """Compute quality score and flag issues."""
    flags: list[str] = []
    cfg = audio_config().get("quality", {})

    clipping = float(np.mean(np.abs(audio) > 0.99)) if len(audio) else 0.0
    if clipping > float(cfg.get("max_clipping_ratio", 0.05)):
        flags.append("clipping")
    if metadata.silence_ratio > float(cfg.get("max_silence_ratio", 0.85)):
        flags.append("excessive_silence")
    if metadata.duration_seconds <= 0:
        flags.append("zero_duration")

    noise_floor = float(np.percentile(np.abs(audio), 10)) if len(audio) else 0.0
    signal = metadata.rms_energy
    snr = 20 * np.log10((signal + 1e-8) / (noise_floor + 1e-8))
    metadata.snr_original = round(float(snr), 2)

    score = 1.0
    score -= min(0.4, clipping * 4)
    score -= min(0.3, max(0.0, metadata.silence_ratio - 0.5))
    score = max(0.0, min(1.0, score))
    metadata.quality_score = round(score, 4)
    return metadata.quality_score, flags
