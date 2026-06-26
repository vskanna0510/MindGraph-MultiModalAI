"""Quality assurance scoring per participant."""

from __future__ import annotations

import pandas as pd

from ml_pipeline.data_quality.config import data_quality_config
from ml_pipeline.data_quality.types import GateResult, GateStatus, QualityScore, ValidationIssue


def categorize(score: float) -> str:
    cfg = data_quality_config().get("quality", {})
    if score >= cfg.get("excellent_threshold", 0.85):
        return "excellent"
    if score >= cfg.get("good_threshold", 0.70):
        return "good"
    if score >= cfg.get("acceptable_threshold", 0.50):
        return "acceptable"
    if score >= cfg.get("reject_threshold", 0.30):
        return "poor"
    return "rejected"


def score_participant(row: pd.Series) -> QualityScore:
    audio = float(row.get("quality_score", 0.8) or 0.8) if row.get("has_audio", True) else 0.0
    video = float(row.get("quality_score", 0.8) or 0.8) if row.get("has_visual", False) else 0.0
    text = float(row.get("quality_score", 0.8) or 0.8) if row.get("has_transcript", True) else 0.0
    sync = float(row.get("sync_score", 1.0) or 1.0)
    meta = 0.9 if row.get("participant_id") else 0.5
    overall = 0.25 * audio + 0.2 * video + 0.25 * text + 0.15 * sync + 0.15 * meta
    return QualityScore(
        audio=round(audio, 4),
        video=round(video, 4),
        transcript=round(text, 4),
        synchronization=round(sync, 4),
        embedding=round(overall, 4),
        metadata=round(meta, 4),
        overall=round(overall, 4),
        category=categorize(overall),
    )


def assess_quality(sessions_df: pd.DataFrame) -> tuple[GateResult, pd.DataFrame]:
    issues: list[ValidationIssue] = []
    reject_below = float(data_quality_config().get("data_quality", {}).get("auto_reject_quality_below", 0.30))
    scores: list[dict] = []

    for _, row in sessions_df.iterrows():
        qs = score_participant(row)
        scores.append(
            {
                "participant_id": row["participant_id"],
                "overall": qs.overall,
                "category": qs.category,
                "audio": qs.audio,
                "video": qs.video,
                "transcript": qs.transcript,
                "sync": qs.synchronization,
            }
        )
        if qs.overall < reject_below:
            issues.append(
                ValidationIssue(
                    "quality_reject",
                    "warning",
                    f"Quality below threshold ({qs.overall})",
                    str(row["participant_id"]),
                )
            )

    rejected = sum(1 for s in scores if s["category"] == "rejected")
    status = GateStatus.PASSED if rejected < len(scores) * 0.5 else GateStatus.WARNING
    return GateResult("quality", status, issues, {"rejected": rejected, "total": len(scores)}), pd.DataFrame(scores)
