"""Tests for video preprocessing pipeline (MP2 Part 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.video.features.facial import extract_facial_features
from ml_pipeline.video.loaders.dataloader import collate_video_batch
from ml_pipeline.video.loaders.datasets import DVLOGVisualDataset
from ml_pipeline.video.normalization.normalize import normalize_embedding, normalize_landmarks
from ml_pipeline.video.pipeline import VideoPipeline
from ml_pipeline.video.preprocess.validation import extract_metadata, verify_integrity
from ml_pipeline.video.sampling.frames import extract_frames, sample_frames
from ml_pipeline.video.tracking.tracker import track_landmarks
from ml_pipeline.video.types import LandmarkFrame


@pytest.fixture
def sample_npy_video(tmp_path: Path) -> Path:
    path = tmp_path / "visual.npy"
    np.save(path, np.random.rand(5, 64, 64).astype(np.float32))
    return path


@pytest.fixture
def dvlog_layout(tmp_path: Path, sample_npy_video: Path) -> Path:
    root = tmp_path / "dvlog" / "0"
    root.mkdir(parents=True)
    dest = root / "0_visual.npy"
    dest.write_bytes(sample_npy_video.read_bytes())
    return tmp_path / "dvlog"


@pytest.fixture
def patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = {
        "processed": tmp_path / "processed",
        "reports": tmp_path / "reports",
        "cache": tmp_path / "cache",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("ml_pipeline.video.config.video_paths", lambda: paths)
    monkeypatch.setattr("ml_pipeline.video.sampling.frames.video_config_module.video_paths", lambda: paths)
    return paths


def test_verify_integrity(sample_npy_video: Path):
    ok, errs = verify_integrity(sample_npy_video)
    assert ok
    assert not errs


def test_extract_metadata(sample_npy_video: Path):
    meta = extract_metadata(sample_npy_video, "0")
    assert meta.frame_count == 5
    assert meta.codec == "npy"


def test_frame_extraction(sample_npy_video: Path, patched_paths):
    frames = extract_frames(sample_npy_video, "0")
    assert len(frames) == 5
    sampled = sample_frames(frames, 1.0)
    assert len(sampled) >= 1


def test_facial_features():
    landmarks = [
        LandmarkFrame(i, float(i), np.random.randn(468, 3).astype(np.float32))
        for i in range(3)
    ]
    features = extract_facial_features(landmarks)
    assert len(features.ear) == 3
    assert features.blink_rate >= 0


def test_tracking():
    landmarks = [LandmarkFrame(0, 0.0, np.zeros((468, 3), dtype=np.float32))]
    tracks = track_landmarks(landmarks)
    assert len(tracks) == 1


def test_normalize():
    arr = np.random.randn(10, 468, 3).astype(np.float32)
    normed = normalize_landmarks(arr)
    assert normed.shape == arr.shape
    vec = np.random.randn(768).astype(np.float32)
    nvec = normalize_embedding(vec)
    assert nvec.shape == vec.shape


def test_collate():
    batch = [
        {"sequence": np.ones((3, 8)), "participant_id": "1"},
        {"sequence": np.ones((5, 8)), "participant_id": "2"},
    ]
    out = collate_video_batch(batch)
    assert out["sequences"].shape == (2, 5, 8)


def test_full_pipeline(sample_npy_video: Path, patched_paths):
    pipeline = VideoPipeline()
    result = pipeline.process_file(sample_npy_video, "0", "0", "test")
    assert result.metadata is not None
    assert len(result.stages) >= 15
    assert result.facial_features is not None


def test_dvlog_discover(dvlog_layout: Path):
    ds = DVLOGVisualDataset(dvlog_layout)
    records = ds.discover()
    assert len(records) == 1
    assert records[0]["participant_id"] == "0"
