"""Tests for dataset parsers (Master Prompt 2 Part 2)."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.datasets.parsers.daic import DAICParser
from ml_pipeline.datasets.parsers.dvlog import DVLOGParser
from ml_pipeline.datasets.parsers.export_index import export_index
from ml_pipeline.datasets.parsers.file_discovery import (
    classify_file,
    discover_files_in_directory,
    validate_audio_file,
)
from ml_pipeline.datasets.parsers.labels import LabelValidationError, OfficialLabelParser
from ml_pipeline.datasets.parsers.quality import below_threshold, score_participant
from ml_pipeline.datasets.parsers.splits import validate_splits, write_split_report
from ml_pipeline.datasets.parsers.statistics import compute_statistics
from ml_pipeline.datasets.parsers.sync import compute_sync
from ml_pipeline.datasets.parsers.transcript import clean_text, parse_transcript
from ml_pipeline.datasets.parsers.types import ParticipantMetadata, ParserResult, SplitInfo, Utterance
from ml_pipeline.datasets.parsers.visualizations import generate_visualizations
from ml_pipeline.datasets.parsers.corruption import detect_corruption
from ml_pipeline.datasets.parsers.report import generate_report
from ml_pipeline.datasets.types import LabelClass


@pytest.fixture
def daic_raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "daic_woz"
    pdir = root / "300_P"
    pdir.mkdir(parents=True)
    wav = pdir / "300_AUDIO.wav"
    with wave.open(str(wav), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)
    with (pdir / "300_TRANSCRIPT.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["start_time", "stop_time", "speaker", "value"])
        writer.writerow([0.0, 1.5, "Ellie", "hello there"])
        writer.writerow([1.6, 3.0, "Participant", "hi"])
    with (root / "train_split_Depression_AVEC2017.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Participant_ID", "PHQ8_Binary", "Gender"])
        writer.writeheader()
        writer.writerow({"Participant_ID": "300", "PHQ8_Binary": "1", "Gender": "1"})
    return root


@pytest.fixture
def dvlog_raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "dvlog"
    (root / "0").mkdir(parents=True)
    np.save(root / "0" / "0_acoustic.npy", np.ones(10))
    np.save(root / "0" / "0_visual.npy", np.ones((5, 10)))
    with (root / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "label", "duration", "gender", "fold"])
        writer.writeheader()
        writer.writerow({"index": "0", "label": "depression", "duration": "100.0", "gender": "f", "fold": "train"})
    return root


@pytest.fixture
def patched_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MINDGRAPH_DATASETS_ROOT", str(tmp_path))
    for sub in ["metadata", "validation", "reports/statistics", "processed/transcripts", "reports"]:
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)


def test_classify_file():
    assert classify_file(Path("x_AUDIO.wav")) == "audio"
    assert classify_file(Path("x_TRANSCRIPT.csv")) == "transcript"
    assert classify_file(Path("x_CLNF_features.txt")) == "facial_features"


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("test\x00bad") == "testbad"


def test_parse_transcript(daic_raw: Path):
    path = daic_raw / "300_P" / "300_TRANSCRIPT.csv"
    data = parse_transcript(path)
    assert data.word_count >= 2
    assert len(data.utterances) == 2
    assert data.quality_score > 0


def test_official_label_parser(daic_raw: Path):
    parser = OfficialLabelParser(daic_raw)
    records, _ = parser.parse()
    assert len(records) == 1
    assert records[0].label == LabelClass.DEPRESSION


def test_label_validation_error(tmp_path: Path):
    root = tmp_path / "daic"
    root.mkdir()
    for name, label in [("train_split.csv", "1"), ("dev_split.csv", "0")]:
        with (root / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Participant_ID", "PHQ8_Binary"])
            writer.writeheader()
            writer.writerow({"Participant_ID": "300", "PHQ8_Binary": label})
    with pytest.raises(LabelValidationError):
        OfficialLabelParser(root).parse()


def test_split_validation():
    records = [
        SplitInfo("1", "train", LabelClass.NORMAL),
        SplitInfo("2", "dev", LabelClass.DEPRESSION),
    ]
    issues, summary = validate_splits(records, {"1", "2", "3"})
    assert summary["train_count"] == 1
    assert any(i.participant_id == "3" for i in issues)


def test_daic_parser_full(daic_raw: Path, patched_env):
    parser = DAICParser(daic_raw)
    result = parser.parse()
    assert len(result.participants) == 1
    assert len(result.samples) >= 2
    assert result.participants[0].label == LabelClass.DEPRESSION
    verify = parser.verify(result)
    assert verify["sample_count"] >= 2


def test_dvlog_parser_full(dvlog_raw: Path, patched_env):
    parser = DVLOGParser(dvlog_raw)
    result = parser.parse()
    assert len(result.participants) == 1
    assert len(result.samples) == 2
    assert result.participants[0].label == LabelClass.DEPRESSION


def test_export_index(patched_env, tmp_path: Path):
    samples = [{"sample_id": "1", "participant_id": "1", "label": "normal"}]
    outputs = export_index(samples, tmp_path / "metadata")
    assert outputs["csv"].exists()
    assert outputs["json"].exists()


def test_sync_and_quality():
    meta = ParticipantMetadata("1", "1", "daic_woz", audio_length=100.0, video_length=98.0)
    from ml_pipeline.datasets.parsers.types import TranscriptData

    td = TranscriptData(utterances=[Utterance("a", 0, 99, "hi", 0)])
    offset, score = compute_sync(meta, td)
    assert offset is not None
    scores = score_participant(meta, td, True, True, True)
    assert scores["overall"] > 0
    assert isinstance(below_threshold(scores), bool)


def test_corruption_detection():
    meta = ParticipantMetadata(
        "1",
        "1",
        "daic_woz",
        audio_length=0,
        files={"transcript": Path("x.csv")},
        transcript_length=0,
    )
    issues = detect_corruption(meta)
    assert any(i.check == "corruption" for i in issues)


def test_statistics_and_report(daic_raw: Path, patched_env, tmp_path: Path):
    parser = DAICParser(daic_raw)
    result = parser.parse()
    stats = compute_statistics(result)
    assert stats["participant_count"] == 1
    path = tmp_path / "reports" / "test_report.md"
    generate_report(result, stats, path)
    assert path.exists()


def test_visualizations(patched_env):
    stats = {"label_distribution": {"normal": 5, "depression": 3}, "participant_count": 8, "average_duration": 100}
    from ml_pipeline.datasets.config import dataset_paths

    viz = generate_visualizations(stats, dataset_paths()["root"] / "reports" / "statistics")
    assert isinstance(viz, list)


def test_discover_files(daic_raw: Path):
    files = discover_files_in_directory(daic_raw / "300_P")
    modalities = {f.modality for f in files}
    assert "audio" in modalities
    assert "transcript" in modalities


def test_validate_audio(daic_raw: Path):
    wav = daic_raw / "300_P" / "300_AUDIO.wav"
    meta, errors = validate_audio_file(wav)
    assert meta["duration_seconds"] > 0
    assert not errors


def test_write_split_report(tmp_path: Path):
    path = write_split_report({"train_count": 1, "valid": True, "unassigned": []}, tmp_path / "split_report.md")
    assert path.exists()
