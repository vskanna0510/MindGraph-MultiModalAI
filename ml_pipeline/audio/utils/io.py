"""Audio I/O and hashing utilities."""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def audio_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as float32 mono array."""
    with wave.open(str(path), "rb") as wav:
        sr = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def load_audio(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio via librosa when available, else wave."""
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {path.suffix}")
    try:
        import librosa

        audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return audio.astype(np.float32), int(sr)
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise
        audio, sr = load_wav_mono(path)
        return audio, sr


def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
