"""Stage 7 — Noise reduction."""

from __future__ import annotations

import numpy as np

from ml_pipeline.audio.config import audio_config


def reduce_noise(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict]:
    """Spectral gating noise reduction with SNR tracking."""
    cfg = audio_config().get("noise_reduction", {})
    if not cfg.get("enabled", True):
        return audio, {"method": "none"}

    noise_floor = np.percentile(np.abs(audio), 15)
    signal_rms = np.sqrt(np.mean(audio**2))
    snr_before = 20 * np.log10((signal_rms + 1e-8) / (noise_floor + 1e-8))

    try:
        import librosa

        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        noise_profile = np.percentile(magnitude, 20, axis=1, keepdims=True)
        mask = magnitude > noise_profile * 1.5
        cleaned = librosa.istft(mask * magnitude * np.exp(1j * phase))
        method = "spectral_gating"
    except ImportError:
        mask = np.abs(audio) > noise_floor * 2
        cleaned = audio * mask
        method = "energy_gate"

    cleaned_rms = np.sqrt(np.mean(cleaned**2))
    snr_after = 20 * np.log10((cleaned_rms + 1e-8) / (noise_floor + 1e-8))
    return cleaned.astype(np.float32), {
        "method": method,
        "snr_original": round(float(snr_before), 2),
        "snr_processed": round(float(snr_after), 2),
        "noise_estimate": round(float(noise_floor), 6),
    }
