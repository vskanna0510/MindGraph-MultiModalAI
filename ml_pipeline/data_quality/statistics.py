"""Statistical validation and distribution analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.data_quality.types import GateResult, GateStatus


def compute_statistics(sessions_df: pd.DataFrame) -> dict:
    stats: dict = {}
    if sessions_df.empty:
        return stats

    for col in ("audio_length", "transcript_length", "quality_score", "sync_score"):
        if col in sessions_df.columns:
            series = pd.to_numeric(sessions_df[col], errors="coerce").dropna()
            if len(series):
                stats[col] = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "variance": float(series.var()),
                    "std": float(series.std()),
                    "skewness": float(series.skew()) if len(series) > 2 else 0.0,
                    "kurtosis": float(series.kurt()) if len(series) > 3 else 0.0,
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                }

    if "label" in sessions_df.columns:
        counts = sessions_df["label"].value_counts()
        total = counts.sum() or 1
        minority = counts.min() if len(counts) else 0
        majority = counts.max() if len(counts) else 0
        stats["class_distribution"] = {
            "counts": counts.to_dict(),
            "imbalance_ratio": float(minority / majority) if majority else 0.0,
            "minority_pct": float(minority / total),
            "majority_pct": float(majority / total),
            "recommended_sampling": "weighted_sampler" if minority / total < 0.15 else "random",
        }

    if "participant_id" in sessions_df.columns:
        stats["participants"] = {
            "count": int(sessions_df["participant_id"].nunique()),
            "sessions": len(sessions_df),
            "avg_duration": float(sessions_df.get("audio_length", pd.Series([0])).mean()),
            "max_duration": float(sessions_df.get("audio_length", pd.Series([0])).max()),
            "min_duration": float(sessions_df.get("audio_length", pd.Series([0])).min()),
        }

    if "language" in sessions_df.columns:
        stats["language_distribution"] = sessions_df["language"].value_counts().to_dict()

    return stats


def validate_statistics(stats: dict) -> GateResult:
    issues = []
    if not stats:
        return GateResult("statistics", GateStatus.WARNING, [], {"empty": True})
    return GateResult("statistics", GateStatus.PASSED, issues, stats)


def write_statistics_report(stats: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    json_path = output_dir / "statistics_report.json"
    json_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    paths["json"] = json_path
    md_lines = ["# Statistics Report\n"]
    for k, v in stats.items():
        md_lines.append(f"## {k}\n\n{v}\n\n")
    md_path = output_dir / "statistics_report.md"
    md_path.write_text("".join(md_lines), encoding="utf-8")
    paths["md"] = md_path
    return paths
