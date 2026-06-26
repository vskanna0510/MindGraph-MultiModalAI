"""Tests for dataset pipeline (Master Prompt 2 Part 1)."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.datasets.adapters.daic_woz import DaicWozAdapter
from ml_pipeline.datasets.adapters.dvlog import DvlogAdapter
from ml_pipeline.datasets.discovery import discover_all, write_dataset_index
from ml_pipeline.datasets.labels import label_to_int, normalize_label
from ml_pipeline.datasets.pipeline import DatasetPipeline
from ml_pipeline.datasets.registry import get_adapter, list_adapters
from ml_pipeline.datasets.types import LabelClass


@pytest.fixture
def synthetic_daic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create minimal DAIC-WOZ raw layout."""
    raw = tmp_path / "raw" / "daic_woz"
    participant = raw / "300_P" / "participant_data" / "audio"
    participant.mkdir(parents=True)
    wav_path = participant / "session.wav"
    with wave.open(str(wav_path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    labels_path = raw / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Participant_ID", "PHQ8_Binary"])
        writer.writeheader()
        writer.writerow({"Participant_ID": "300", "PHQ8_Binary": "1"})

    splits = raw / "splits"
    splits.mkdir()
    with (splits / "train_split.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Participant_ID"])
        writer.writeheader()
        writer.writerow({"Participant_ID": "300"})

    monkeypatch.setenv("MINDGRAPH_DATASETS_ROOT", str(tmp_path))
    return raw


@pytest.fixture
def synthetic_dvlog(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "dvlog"
    raw.mkdir(parents=True)
    np.save(raw / "acoustic_001.npy", np.array([1.0, 2.0, 3.0]))
    with (raw / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["participant", "label"])
        writer.writeheader()
        writer.writerow({"participant": "001", "label": "0"})
    return raw


@pytest.fixture
def patched_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all dataset paths to tmp_path via environment."""
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    for sub in ["processed", "cache", "metadata", "exports", "logs", "validation", "splits", "statistics"]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MINDGRAPH_DATASETS_ROOT", str(tmp_path))
    from ml_pipeline.datasets.config import dataset_paths

    return dataset_paths()


def test_list_adapters():
    assert "daic_woz" in list_adapters()
    assert "dvlog" in list_adapters()


def test_normalize_label():
    assert normalize_label("1") == LabelClass.DEPRESSION
    assert normalize_label("normal") == LabelClass.NORMAL
    assert normalize_label("") == LabelClass.UNKNOWN


def test_label_to_int_from_config():
    assert label_to_int(LabelClass.NORMAL) == 0
    assert label_to_int(LabelClass.DEPRESSION) == 1


def test_daic_adapter_discover(synthetic_daic: Path):
    adapter = DaicWozAdapter(synthetic_daic)
    records = adapter.discover()
    assert len(records) == 1
    assert records[0].participant_id == "300"
    assert records[0].label == LabelClass.DEPRESSION


def test_dvlog_adapter_discover(synthetic_dvlog: Path):
    adapter = DvlogAdapter(synthetic_dvlog)
    records = adapter.discover()
    assert len(records) == 1
    assert records[0].modality == "audio"


def test_full_pipeline(synthetic_daic: Path, patched_paths, monkeypatch: pytest.MonkeyPatch):
    from ml_pipeline.datasets import config, registry

    monkeypatch.setattr(
        "ml_pipeline.datasets.discovery.registry.get_adapter",
        lambda name, raw_root=None: DaicWozAdapter(synthetic_daic),
    )
    monkeypatch.setattr(
        "ml_pipeline.datasets.discovery.registry.list_adapters",
        lambda: ["daic_woz"],
    )

    pipeline = DatasetPipeline(datasets=["daic_woz"])
    result = pipeline.run()
    assert "discovery" in result.steps_completed
    assert "verification" in result.steps_completed
    assert len(result.records) == 1
    assert (patched_paths["metadata"] / "dataset_index.csv").exists()
    assert (patched_paths["metadata"] / "manifest.json").exists()
    assert result.verification.get("passed") is True


def test_write_dataset_index(synthetic_daic: Path, patched_paths):
    adapter = DaicWozAdapter(synthetic_daic)
    records = adapter.discover()
    out = write_dataset_index(records, patched_paths["metadata"] / "dataset_index.csv")
    assert out.exists()
