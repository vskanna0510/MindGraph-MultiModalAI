"""Data quality engine."""

from __future__ import annotations

from ml_pipeline.feature_store.config import feature_store_config
from ml_pipeline.feature_store.types import MultimodalSession, QualityCategory, QualityMetrics


def categorize_score(score: float) -> QualityCategory:
    cfg = feature_store_config().get("quality", {})
    if score >= cfg.get("excellent_threshold", 0.85):
        return QualityCategory.EXCELLENT
    if score >= cfg.get("good_threshold", 0.70):
        return QualityCategory.GOOD
    if score >= cfg.get("acceptable_threshold", 0.50):
        return QualityCategory.ACCEPTABLE
    if score >= cfg.get("reject_threshold", 0.30):
        return QualityCategory.POOR
    return QualityCategory.REJECTED


def assess_session_quality(session: MultimodalSession) -> QualityMetrics:
    audio_q = session.quality.audio if session.modality_mask.get("audio") else 0.0
    video_q = session.quality.video if session.modality_mask.get("visual") else 0.0
    text_q = session.quality.text if session.modality_mask.get("text") else 0.0
    sync_q = session.sync_score
    meta_q = 1.0 if session.metadata else 0.5
    present = sum(1 for v in session.modality_mask.values() if v)
    modality_bonus = present / max(len(session.modality_mask), 1)
    overall = 0.25 * audio_q + 0.2 * video_q + 0.25 * text_q + 0.15 * sync_q + 0.15 * meta_q
    overall = min(1.0, overall * (0.5 + 0.5 * modality_bonus))
    return QualityMetrics(
        audio=audio_q,
        video=video_q,
        text=text_q,
        synchronization=sync_q,
        embedding=overall,
        metadata=meta_q,
        overall=round(overall, 4),
        category=categorize_score(overall),
    )
