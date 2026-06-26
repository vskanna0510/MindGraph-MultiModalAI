"""Stage 6 — Amplitude normalization."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config


def normalize_amplitude(audio: np.ndarray, method: str | None = None) -> np.ndarray:
    """Peak, RMS, or LUFS-style normalization."""
    cfg = audio_config().get("amplitude", {})
    method = method or cfg.get("method", "peak")
    if len(audio) == 0:
        return audio
    if method == "rms":
        rms = np.sqrt(np.mean(audio**2)) + 1e-8
        return (audio / rms * float(cfg.get("target_rms", 0.1))).astype(np.float32)
    if method == "lufs":
        rms = np.sqrt(np.mean(audio**2)) + 1e-8
        target = 10 ** (float(cfg.get("target_lufs", -23.0)) / 20) * 0.1
        return (audio / rms * target).astype(np.float32)
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak * float(cfg.get("target_peak", 0.95))).astype(np.float32)
