"""Tests for data quality engine (MP2 Part 7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.data_quality.analysis import correlation_matrix, detect_outliers
from ml_pipeline.data_quality.audit import build_audit_entry
from ml_pipeline.data_quality.config import data_quality_config, data_quality_paths
from ml_pipeline.data_quality.consistency import validate_consistency
from ml_pipeline.data_quality.embeddings import validate_embeddings
from ml_pipeline.data_quality.integrity import validate_integrity, write_integrity_report
from ml_pipeline.data_quality.leakage import detect_leakage, write_leakage_report
from ml_pipeline.data_quality.passport import build_passport, save_passport
from ml_pipeline.data_quality.pipeline import DataQualityPipeline
from ml_pipeline.data_quality.quality_engine import assess_quality, categorize, score_participant
from ml_pipeline.data_quality.reports import generate_master_report, write_reports
from ml_pipeline.data_quality.statistics import compute_statistics, validate_statistics, write_statistics_report
from ml_pipeline.data_quality.types import DataQualityResult, GateResult, GateStatus, ValidationIssue


@pytest.fixture
def sessions_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "participant_id": "300",
                "session_id": "300",
                "dataset": "daic_woz",
                "split": "train",
                "label": "normal",
                "language": "en",
                "has_audio": True,
                "has_visual": False,
                "has_transcript": True,
                "audio_length": 100.0,
                "transcript_length": 500,
                "quality_score": 0.85,
                "sync_score": 0.9,
            },
            {
                "participant_id": "301",
                "session_id": "301",
                "dataset": "daic_woz",
                "split": "train",
                "label": "depression",
                "language": "en",
                "has_audio": True,
                "has_visual": False,
                "has_transcript": True,
                "audio_length": 80.0,
                "transcript_length": 400,
                "quality_score": 0.75,
                "sync_score": 0.85,
            },
        ]
    )


@pytest.fixture
def index_df(tmp_path: Path) -> pd.DataFrame:
    f1 = tmp_path / "a.wav"
    f1.write_bytes(b"wav")
    f2 = tmp_path / "b.csv"
    f2.write_text("a,b\n1,2", encoding="utf-8")
    return pd.DataFrame(
        [
            {"participant_id": "300", "modality": "audio", "file_path": str(f1)},
            {"participant_id": "300", "modality": "transcript", "file_path": str(f2)},
        ]
    )


def test_config():
    assert data_quality_config().get("data_quality", {}).get("version")
    assert "reports" in data_quality_paths()


def test_integrity_validation(index_df: pd.DataFrame):
    cfg = data_quality_config()
    result = validate_integrity(index_df, cfg)
    assert result.status == GateStatus.PASSED


def test_integrity_missing_file(tmp_path: Path):
    df = pd.DataFrame([{"participant_id": "1", "modality": "audio", "file_path": str(tmp_path / "missing.wav")}])
    result = validate_integrity(df, data_quality_config())
    assert result.status == GateStatus.FAILED


def test_consistency_validation(sessions_df: pd.DataFrame):
    result = validate_consistency(sessions_df)
    assert result.status == GateStatus.PASSED


def test_leakage_detection(sessions_df: pd.DataFrame, index_df: pd.DataFrame):
    leaked = pd.concat([sessions_df, sessions_df.iloc[[0]].assign(split="test")], ignore_index=True)
    result = detect_leakage(leaked, index_df)
    assert result.status == GateStatus.FAILED


def test_no_leakage(sessions_df: pd.DataFrame, index_df: pd.DataFrame):
    result = detect_leakage(sessions_df, index_df)
    assert result.status == GateStatus.PASSED


def test_quality_scoring(sessions_df: pd.DataFrame):
    row = sessions_df.iloc[0]
    score = score_participant(row)
    assert score.overall > 0
    assert categorize(0.9) == "excellent"
    gate, qdf = assess_quality(sessions_df)
    assert len(qdf) == 2


def test_statistics(sessions_df: pd.DataFrame):
    stats = compute_statistics(sessions_df)
    assert "class_distribution" in stats
    gate = validate_statistics(stats)
    assert gate.status == GateStatus.PASSED


def test_outlier_detection():
    series = pd.Series([1.0, 1.0, 1.0, 1.0, 50.0])
    flags = detect_outliers(series, method="isolation_forest")
    assert len(flags) == 5


def test_correlation(sessions_df: pd.DataFrame):
    corr = correlation_matrix(sessions_df, ["audio_length", "quality_score"])
    assert corr.shape == (2, 2)


def test_passport(sessions_df: pd.DataFrame):
    gates = [GateResult("integrity", GateStatus.PASSED)]
    passport = build_passport(sessions_df, gates, True)
    assert passport.participants == 2
    assert passport.validation_status == "approved"


def test_reports(tmp_path: Path, sessions_df: pd.DataFrame):
    result = DataQualityResult(
        gates=[GateResult("integrity", GateStatus.PASSED)],
        approved=True,
    )
    passport = build_passport(sessions_df, result.gates, True)
    stats = compute_statistics(sessions_df)
    paths = write_reports(result, passport, stats, tmp_path)
    assert paths["markdown"].exists()
    assert "Approved" in generate_master_report(result, passport, stats)


def test_audit_entry():
    result = DataQualityResult(gates=[GateResult("test", GateStatus.PASSED)], approved=True)
    audit = build_audit_entry(result, "start", "abc", "1.0")
    assert audit.approval_status.value == "approved"


def test_integrity_report(tmp_path: Path):
    result = GateResult("integrity", GateStatus.PASSED)
    path = write_integrity_report(result, tmp_path / "integrity_report.md")
    assert path.exists()


def test_leakage_report(tmp_path: Path):
    result = GateResult("leakage", GateStatus.PASSED)
    path = write_leakage_report(result, tmp_path / "leakage.md")
    assert path.exists()


def test_statistics_report(tmp_path: Path, sessions_df: pd.DataFrame):
    stats = compute_statistics(sessions_df)
    paths = write_statistics_report(stats, tmp_path)
    assert paths["json"].exists()


def test_embedding_validation(tmp_path: Path, sessions_df: pd.DataFrame):
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    arr = np.random.randn(128).astype(np.float32)
    np.save(emb_dir / "300_300_text.npy", arr)
    result = validate_embeddings(sessions_df, [emb_dir])
    assert result.metadata["embeddings_found"] >= 1


def test_pipeline_monkeypatch(sessions_df: pd.DataFrame, index_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "ml_pipeline.data_quality.pipeline.DataQualityPipeline._load_sessions",
        lambda self: sessions_df,
    )
    monkeypatch.setattr(
        "ml_pipeline.data_quality.pipeline.DataQualityPipeline._load_index",
        lambda self: index_df,
    )
    monkeypatch.setattr(
        "ml_pipeline.data_quality.pipeline.data_quality_paths",
        lambda: {
            "reports": tmp_path / "reports",
            "figures": tmp_path / "figures",
            "audit_log": tmp_path / "audit.jsonl",
            "passport": tmp_path / "passport.json",
            "ci_report": tmp_path / "ci.json",
            "dataset_index": tmp_path / "index.csv",
            "feature_sessions": tmp_path / "sessions.csv",
        },
    )
    pipeline = DataQualityPipeline()
    result = pipeline.run()
    assert len(result.gates) >= 5
    assert result.passport is not None
    assert (tmp_path / "passport.json").exists()
