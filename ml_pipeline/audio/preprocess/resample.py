"""Stage 4 — Resampling."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config


def resample(audio: np.ndarray, source_sr: int, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Resample audio to target sample rate."""
    cfg = audio_config().get("audio", {})
    target = target_sr or int(cfg.get("target_sample_rate", 16000))
    if source_sr == target:
        return audio, source_sr
    try:
        import librosa

        return librosa.resample(audio, orig_sr=source_sr, target_sr=target).astype(np.float32), target
    except ImportError:
        ratio = target / source_sr
        indices = np.arange(0, len(audio), 1 / ratio)
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices].astype(np.float32), target
