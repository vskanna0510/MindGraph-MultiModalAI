"""Dataset passport generation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_pipeline.data_quality.config import data_quality_config
from ml_pipeline.data_quality.reproducibility import get_git_commit, config_hash
from ml_pipeline.data_quality.types import ApprovalStatus, DatasetPassport, GateResult


def build_passport(
    sessions_df: pd.DataFrame,
    gates: list[GateResult],
    approved: bool,
) -> DatasetPassport:
    cfg = data_quality_config()
    dq = cfg.get("data_quality", {})
    content = sessions_df.to_json() if not sessions_df.empty else ""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    avg_quality = float(sessions_df["quality_score"].mean()) if "quality_score" in sessions_df.columns and len(sessions_df) else 0.0

    return DatasetPassport(
        dataset_name="MindGraph-MultiDep",
        version=dq.get("dataset_version", "v1"),
        hash=content_hash,
        creation_date=datetime.now(UTC).isoformat(),
        source="DAIC-WOZ,D-VLOG",
        license="Research-only",
        num_samples=len(sessions_df),
        participants=int(sessions_df["participant_id"].nunique()) if not sessions_df.empty else 0,
        sessions=len(sessions_df),
        languages=sessions_df["language"].unique().tolist() if "language" in sessions_df.columns else ["en"],
        modalities=["audio", "visual", "text"],
        validation_status="approved" if approved else "rejected",
        quality_score=round(avg_quality, 4),
        processing_version=dq.get("pipeline_version", "1.0.0"),
        git_commit=get_git_commit(),
        random_seed=int(cfg.get("reproducibility", {}).get("random_seed", 42)),
        feature_version="v1.0",
        embedding_version="v1.0",
    )


def save_passport(passport: DatasetPassport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(passport.__dict__, indent=2), encoding="utf-8")
    return path
