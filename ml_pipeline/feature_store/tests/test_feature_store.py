"""Tests for multimodal feature store (MP2 Part 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.feature_store.balancing import class_weights, compute_class_distribution
from ml_pipeline.feature_store.cache import FeatureCache
from ml_pipeline.feature_store.collate import collate_multimodal_batch
from ml_pipeline.feature_store.config import feature_store_config, feature_store_paths
from ml_pipeline.feature_store.datasets.combined import CombinedDataset
from ml_pipeline.feature_store.datasets.daic import DAICDataset
from ml_pipeline.feature_store.index import build_session_index, sessions_to_multimodal
from ml_pipeline.feature_store.normalization import apply_normalization, fit_zscore
from ml_pipeline.feature_store.outlier import detect_outliers_zscore, detect_embedding_outliers
from ml_pipeline.feature_store.quality import assess_session_quality, categorize_score
from ml_pipeline.feature_store.registry import FeatureRegistry
from ml_pipeline.feature_store.reproducibility import build_reproducibility_manifest, config_hash
from ml_pipeline.feature_store.statistics import compute_dataset_statistics
from ml_pipeline.feature_store.store import FeatureStore, checksum_array
from ml_pipeline.feature_store.sync import compute_sync_score
from ml_pipeline.feature_store.types import FeatureRecord, FeatureType, MultimodalSession, QualityMetrics, SessionBatch
from ml_pipeline.feature_store.validation import validate_feature, validate_session_batch, write_validation_report


@pytest.fixture
def sample_index_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "participant_id": "300",
                "session_id": "300",
                "dataset": "daic_woz",
                "modality": "audio",
                "file_path": "/data/300_AUDIO.wav",
                "label": "normal",
                "split": "train",
                "language": "en",
                "audio_length": 100.0,
                "transcript_length": 500,
                "quality_score": 0.85,
                "sync_score": 0.9,
            },
            {
                "participant_id": "300",
                "session_id": "300",
                "dataset": "daic_woz",
                "modality": "transcript",
                "file_path": "/data/300_TRANSCRIPT.csv",
                "label": "normal",
                "split": "train",
                "language": "en",
                "audio_length": 100.0,
                "transcript_length": 500,
                "quality_score": 0.85,
                "sync_score": 0.9,
            },
            {
                "participant_id": "1",
                "session_id": "1",
                "dataset": "dvlog",
                "modality": "audio",
                "file_path": "/data/1_acoustic.npy",
                "label": "depression",
                "split": "train",
                "language": "en",
                "audio_length": 30.0,
                "transcript_length": 0,
                "quality_score": 0.7,
                "sync_score": 0.5,
            },
        ]
    )


def test_feature_store_config():
    cfg = feature_store_config()
    paths = feature_store_paths()
    assert "root" in paths
    assert cfg.get("feature_store", {}).get("version")


def test_build_session_index(sample_index_df: pd.DataFrame):
    sessions = build_session_index(sample_index_df)
    assert len(sessions) == 2
    assert sessions.iloc[0]["has_audio"]


def test_sessions_to_multimodal(sample_index_df: pd.DataFrame):
    df = build_session_index(sample_index_df)
    sessions = sessions_to_multimodal(df)
    assert len(sessions) == 2
    assert sessions[0].modality_mask["audio"]


def test_feature_record_and_store(tmp_path: Path):
    arr = np.random.randn(128).astype(np.float32)
    record = FeatureRecord(
        feature_id="300_300_audio_v1",
        participant_id="300",
        session_id="300",
        feature_type=FeatureType.AUDIO,
        feature_version="v1.0",
        extraction_version="1.0.0",
        model_version="test",
        dataset="daic_woz",
        language="en",
        split="train",
        embedding_shape=(128,),
        embedding_dim=128,
        normalization_method="zscore",
        storage_path=tmp_path / "300.npy",
        checksum=checksum_array(arr),
        quality_score=0.9,
        extraction_time="now",
        processing_duration_ms=10.0,
    )
    store = FeatureStore(tmp_path)
    path = store.save_numpy(arr, record)
    assert path.exists()
    loaded = store.load_numpy(path)
    assert loaded.shape == (128,)
    ok, errs = validate_feature(record, loaded)
    assert ok
    assert not errs


def test_validate_session_batch():
    batch = SessionBatch(
        participant_id="300",
        session_id="300",
        audio=np.array([0.1, 0.2], dtype=np.float32),
        visual=None,
        text=np.array([0.5, 0.6], dtype=np.float32),
        image=None,
        graph=None,
        clinical={},
        label=0,
        quality=QualityMetrics(overall=0.8),
        language="en",
        split="train",
        timestamp="now",
        modality_mask={"audio": True, "visual": False, "text": True},
    )
    ok, errs = validate_session_batch(batch)
    assert ok


def test_collate_multimodal_batch():
    batch = [
        {
            "participant_id": "1",
            "session_id": "1",
            "audio": np.array([1.0, 2.0], dtype=np.float32),
            "visual": None,
            "text": np.array([0.1], dtype=np.float32),
            "label": 0,
            "modality_mask": {"audio": True, "visual": False, "text": True},
        },
        {
            "participant_id": "2",
            "session_id": "2",
            "audio": np.array([3.0], dtype=np.float32),
            "visual": None,
            "text": None,
            "label": 1,
            "modality_mask": {"audio": True, "visual": False, "text": False},
        },
    ]
    out = collate_multimodal_batch(batch)
    assert out["audio"].shape[0] == 2
    assert out["audio_mask"].shape == (2, 2)
    assert "modality_presence" in out


def test_normalization():
    arrs = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])]
    params = fit_zscore(arrs)
    normed = apply_normalization(arrs[0], params)
    assert normed is not None
    assert abs(float(np.mean(normed))) < 1.0


def test_quality_engine():
    session = MultimodalSession(
        participant_id="300",
        session_id="300",
        dataset="daic_woz",
        split="train",
        language="en",
        label="normal",
        modality_mask={"audio": True, "visual": False, "text": True},
        quality=QualityMetrics(audio=0.9, text=0.8),
        sync_score=0.95,
    )
    q = assess_session_quality(session)
    assert q.overall > 0
    assert categorize_score(0.9) == categorize_score(0.9)


def test_outlier_detection():
    values = np.array([1.0, 1.0, 1.0, 1.0, 50.0])
    flags = detect_outliers_zscore(values, threshold=1.5)
    assert bool(flags[-1])
    embs = [np.ones(10), np.ones(10), np.ones(10) * 500]
    out = detect_embedding_outliers(embs, method="zscore", threshold=1.0)
    assert len(out) == 3


def test_sync_score():
    score = compute_sync_score(100.0, 1500)
    assert 0 < score <= 1.0


def test_class_balancing():
    dist = compute_class_distribution(["normal", "normal", "depression"])
    assert dist["distribution"]["normal"] == 2
    weights = class_weights([0, 0, 1, 1, 1])
    assert len(weights) >= 2


def test_feature_registry(tmp_path: Path):
    reg = FeatureRegistry(path=tmp_path / "registry.json")
    assert len(reg.entries) > 0
    path = reg.save()
    assert path.exists()


def test_feature_cache(tmp_path: Path):
    cache = FeatureCache(tmp_path, "abc")
    cache.set("300", "300", "v1", "m1", {"path": "/x"})
    hit = cache.get("300", "300", "v1", "m1")
    assert hit["path"] == "/x"
    assert cache.hit_rate > 0


def test_reproducibility_manifest():
    m = build_reproducibility_manifest("abc123")
    assert m.dataset_version
    assert len(config_hash()) == 16


def test_dataset_statistics(sample_index_df: pd.DataFrame):
    sessions = build_session_index(sample_index_df)
    stats = compute_dataset_statistics(sessions)
    assert stats["sessions"] == 2
    assert "class_distribution" in stats


def test_daic_dataset_monkeypatch(sample_index_df: pd.DataFrame, monkeypatch):
    monkeypatch.setattr(
        "ml_pipeline.feature_store.datasets.daic.load_dataset_index",
        lambda: sample_index_df,
    )
    ds = DAICDataset(split="train")
    df = ds.load_metadata()
    assert len(df) == 1
    assert ds.load_labels(df.iloc[0].to_dict()) == 0


def test_combined_dataset_monkeypatch(sample_index_df: pd.DataFrame, monkeypatch):
    monkeypatch.setattr(
        "ml_pipeline.feature_store.datasets.daic.load_dataset_index",
        lambda: sample_index_df,
    )
    monkeypatch.setattr(
        "ml_pipeline.feature_store.datasets.dvlog.load_dataset_index",
        lambda: sample_index_df,
    )
    ds = CombinedDataset()
    records = ds.discover()
    assert len(records) >= 2


def test_validation_report(tmp_path: Path):
    results = [{"participant_id": "300", "passed": True, "errors": []}]
    path = write_validation_report(results, tmp_path / "report.md")
    assert path.exists()
    assert "PASS" in path.read_text(encoding="utf-8")


def test_daic_dataset_getitem_monkeypatch(sample_index_df: pd.DataFrame, monkeypatch, tmp_path: Path):
    audio = tmp_path / "300_AUDIO.wav"
    audio.write_bytes(b"")
    df = sample_index_df.copy()
    df.loc[df["participant_id"] == "300", "file_path"] = str(audio)
    monkeypatch.setattr(
        "ml_pipeline.feature_store.datasets.base.load_dataset_index",
        lambda: df,
    )
    ds = DAICDataset()
    ds.load_metadata()
    item = ds[0]
    assert "participant_id" in item
    assert "modality_mask" in item
