"""Stage 11 — Acoustic feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.audio.config import audio_config, audio_paths
from ml_pipeline.audio.types import FeatureBundle


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    participant_id: str,
) -> FeatureBundle:
    """Extract MFCC, mel, spectral, and prosody features."""
    cfg = audio_config()
    feat_cfg = cfg.get("features", {})
    version = cfg.get("pipeline", {}).get("feature_version", "v1")
    paths = audio_paths()
    out_root = paths["processed"]

    n_mfcc = int(feat_cfg.get("mfcc_coeffs", 13))
    hop = int(feat_cfg.get("hop_length", 512))
    n_fft = int(feat_cfg.get("n_fft", 2048))
    n_mels = int(feat_cfg.get("n_mels", 128))

    bundle = FeatureBundle(participant_id=participant_id, feature_version=version)

    try:
        import librosa

        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop, n_fft=n_fft)
        if feat_cfg.get("include_delta", True):
            delta = librosa.feature.delta(mfcc)
            mfcc = np.vstack([mfcc, delta])
        if feat_cfg.get("include_delta_delta", True):
            ddelta = librosa.feature.delta(mfcc[:n_mfcc], order=2)
            mfcc = np.vstack([mfcc, ddelta])
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, hop_length=hop, n_fft=n_fft)
        log_mel = librosa.power_to_db(mel)
        bundle.mfcc = mfcc
        bundle.mel = log_mel
        bundle.spectral = {
            "centroid": librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop).mean(),
            "bandwidth": librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, hop_length=hop).mean(),
            "rolloff": librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, hop_length=hop).mean(),
            "zcr": librosa.feature.zero_crossing_rate(audio, hop_length=hop).mean(),
            "rms": librosa.feature.rms(y=audio, hop_length=hop).mean(),
        }
        if feat_cfg.get("extract_pitch", True):
            f0, _, _ = librosa.pyin(audio, fmin=50, fmax=400, sr=sample_rate)
            f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            bundle.pitch = {
                "f0_mean": float(np.mean(f0_clean)) if len(f0_clean) else 0.0,
                "f0_std": float(np.std(f0_clean)) if len(f0_clean) else 0.0,
            }
        bundle.prosody = {"speaking_rate": len(audio) / sample_rate}
    except ImportError:
        bundle.mfcc = np.array([np.mean(audio), np.std(audio)])
        bundle.mel = bundle.mfcc

    for name, arr in [("mfcc", bundle.mfcc), ("mel", bundle.mel)]:
        if arr is not None:
            subdir = out_root / name
            subdir.mkdir(parents=True, exist_ok=True)
            path = subdir / f"{participant_id}_{name}_{version}.npy"
            np.save(path, arr)
            bundle.paths[name] = path

    return bundle
