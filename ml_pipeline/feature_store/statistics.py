"""Dataset statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml_pipeline.feature_store.balancing import compute_class_distribution


def compute_dataset_statistics(index_df: pd.DataFrame) -> dict[str, Any]:
    if index_df.empty:
        return {"participants": 0, "sessions": 0}

    labels = index_df["label"].tolist() if "label" in index_df.columns else []
    langs = index_df["language"].value_counts().to_dict() if "language" in index_df.columns else {}
    quality = index_df["quality_score"].describe().to_dict() if "quality_score" in index_df.columns else {}

    missing_audio = int((~index_df["has_audio"]).sum()) if "has_audio" in index_df.columns else 0
    missing_visual = int((~index_df["has_visual"]).sum()) if "has_visual" in index_df.columns else 0
    missing_text = int((~index_df["has_transcript"]).sum()) if "has_transcript" in index_df.columns else 0

    return {
        "participants": int(index_df["participant_id"].nunique()),
        "sessions": len(index_df),
        "avg_audio_length": float(index_df["audio_length"].mean()) if "audio_length" in index_df.columns else 0.0,
        "avg_transcript_length": float(index_df["transcript_length"].mean()) if "transcript_length" in index_df.columns else 0.0,
        "missing_modalities": {
            "audio": missing_audio,
            "visual": missing_visual,
            "text": missing_text,
        },
        "language_distribution": langs,
        "class_distribution": compute_class_distribution(labels),
        "quality_distribution": quality,
    }


def write_statistics(stats: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    json_path = output_dir / "statistics.json"
    json_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    paths["json"] = json_path
    md_path = output_dir / "statistics.md"
    lines = ["# Dataset Statistics\n"]
    for k, v in stats.items():
        lines.append(f"## {k}\n\n{v}\n\n")
    md_path.write_text("".join(lines), encoding="utf-8")
    paths["md"] = md_path
    try:
        flat = pd.json_normalize(stats)
        pq = output_dir / "statistics.parquet"
        flat.to_parquet(pq, index=False)
        paths["parquet"] = pq
    except Exception:
        pass
    return paths
