"""Tests for audio engineering pipeline (MP2 Part 3)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.audio.augmentation.augment import maybe_augment
from ml_pipeline.audio.cache.manager import AudioCacheManager
from ml_pipeline.audio.export.exporter import export_artifacts
from ml_pipeline.audio.features.extractor import extract_features
from ml_pipeline.audio.loaders.dataloader import collate_audio_batch
from ml_pipeline.audio.loaders.datasets import DAICAudioDataset
from ml_pipeline.audio.normalization.feature_norm import normalize_features
from ml_pipeline.audio.pipeline import AudioPipeline
from ml_pipeline.audio.preprocess.amplitude import normalize_amplitude
from ml_pipeline.audio.preprocess.integrity import verify_integrity
from ml_pipeline.audio.preprocess.resample import resample
from ml_pipeline.audio.quality.validation import validate_features
from ml_pipeline.audio.segmentation.vad import run_vad
from ml_pipeline.audio.types import EmbeddingRecord, FeatureBundle
from ml_pipeline.audio.utils.io import save_wav


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    path = tmp_path / "test.wav"
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    save_wav(path, audio, sr)
    return path


@pytest.fixture
def daic_layout(tmp_path: Path, sample_wav: Path) -> Path:
    root = tmp_path / "daic_woz" / "300_P"
    root.mkdir(parents=True)
    dest = root / "300_AUDIO.wav"
    dest.write_bytes(sample_wav.read_bytes())
    return tmp_path / "daic_woz"


def test_verify_integrity(sample_wav: Path):
    ok, errs = verify_integrity(sample_wav)
    assert ok
    assert not errs


def test_verify_integrity_missing(tmp_path: Path):
    ok, errs = verify_integrity(tmp_path / "missing.wav")
    assert not ok
    assert "missing_file" in errs


def test_resample_and_amplitude():
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    resampled, sr = resample(audio, 16000, 16000)
    assert sr == 16000
    normed = normalize_amplitude(resampled)
    assert np.max(np.abs(normed)) <= 1.0


def test_vad(sample_wav: Path):
    from ml_pipeline.audio.utils.io import load_audio

    audio, sr = load_audio(sample_wav)
    vad = run_vad(audio, sr)
    assert 0 <= vad.speech_ratio <= 1


def test_feature_extraction(sample_wav: Path):
    from ml_pipeline.audio.utils.io import load_audio

    audio, sr = load_audio(sample_wav)
    bundle = extract_features(audio, sr, "300")
    assert bundle.mfcc is not None
    valid, errs = validate_features(bundle)
    assert valid or not errs


def test_normalize_features():
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    normed, stats = normalize_features(arr, "zscore")
    assert "mean" in stats
    assert abs(float(normed.mean())) < 0.01


def test_collate_batch():
    batch = [
        {"audio": np.ones(1000, dtype=np.float32), "participant_id": "1", "sample_rate": 16000},
        {"audio": np.ones(500, dtype=np.float32), "participant_id": "2", "sample_rate": 16000},
    ]
    collated = collate_audio_batch(batch)
    assert collated["audio"].shape == (2, 1000)
    assert collated["attention_mask"].shape == (2, 1000)


def test_cache_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ml_pipeline.audio.cache.manager.audio_paths",
        lambda: {"cache": tmp_path / "cache"},
    )
    cache = AudioCacheManager()
    key = cache.cache_key("abc123", "1.0")
    path = tmp_path / "cache" / "test.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.ones(4))
    cache.put(key, path)
    assert cache.get(key) == path


def test_full_pipeline(sample_wav: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "ml_pipeline.audio.config.audio_paths",
        lambda: {
            "processed": tmp_path / "processed",
            "reports": tmp_path / "reports",
            "cache": tmp_path / "cache",
        },
    )
    pipeline = AudioPipeline()
    result = pipeline.process_file(sample_wav, "300", "300", "train")
    assert result.metadata is not None
    assert len(result.stages) >= 14
    assert result.features is not None
    exports = export_artifacts(result)
    assert "numpy" in exports or "pickle" in exports


def test_daic_dataset_discover(daic_layout: Path):
    ds = DAICAudioDataset(daic_layout)
    records = ds.discover()
    assert len(records) == 1
    assert records[0]["participant_id"] == "300"


def test_augmentation_training_only():
    audio = np.ones(1000, dtype=np.float32) * 0.5
    out = maybe_augment(audio, 16000, "train")
    assert out.shape == audio.shape
    unchanged = maybe_augment(audio, 16000, "test")
    np.testing.assert_array_equal(unchanged, audio)


def test_embedding_validation():
    bundle = FeatureBundle(participant_id="1", feature_version="v1", mfcc=np.ones((26, 10)))
    emb = EmbeddingRecord(
        participant_id="1",
        session_id="1",
        model_name="test",
        model_version="1",
        embedding_dim=8,
        vector_path=Path("nonexistent.npy"),
        audio_hash="x",
        config_version="1",
        extraction_timestamp="now",
    )
    valid, errs = validate_features(bundle, emb)
    assert not valid
    assert "embedding_missing" in errs
