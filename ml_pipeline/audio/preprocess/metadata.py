"""Stage 2 — Metadata extraction."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ml_pipeline.audio.config import audio_config
from ml_pipeline.audio.types import AudioMetadata
from ml_pipeline.audio.utils.io import load_audio


def extract_metadata(
    path: Path,
    participant_id: str,
    session_id: str | None = None,
) -> AudioMetadata:
    """Extract duration, sample rate, energy, and related metadata."""
    cfg = audio_config().get("audio", {})
    audio, sr = load_audio(path)
    duration = len(audio) / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0

    channels = 1
    bit_depth = None
    codec = path.suffix.lower().lstrip(".")
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            bit_depth = wav.getsampwidth() * 8
            codec = "pcm"

    silence_mask = np.abs(audio) < 0.01
    silence_ratio = float(np.mean(silence_mask)) if len(audio) else 1.0

    return AudioMetadata(
        participant_id=participant_id,
        session_id=session_id or participant_id,
        source_path=str(path),
        duration_seconds=round(duration, 3),
        sample_rate=sr,
        channels=channels,
        bit_depth=bit_depth,
        codec=codec,
        bitrate=None,
        peak_amplitude=round(peak, 6),
        rms_energy=round(rms, 6),
        silence_ratio=round(silence_ratio, 4),
        speech_ratio=round(1.0 - silence_ratio, 4),
        original_channels=channels,
    )
