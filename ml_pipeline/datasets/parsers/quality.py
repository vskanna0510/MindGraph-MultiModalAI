"""Quality scoring for parsed participants."""

from __future__ import annotations

from ml_pipeline.datasets.config import dataset_config
from ml_pipeline.datasets.parsers.types import ParticipantMetadata, TranscriptData


def score_participant(
    participant: ParticipantMetadata,
    transcript: TranscriptData | None,
    has_audio: bool,
    has_video: bool,
    has_transcript: bool,
) -> dict[str, float]:
    """Generate per-modality and overall quality scores."""
    audio_score = 1.0 if has_audio and participant.audio_length else 0.0
    video_score = 1.0 if has_video and participant.video_length else 0.0
    transcript_score = transcript.quality_score if transcript else (1.0 if has_transcript else 0.0)
    sync_score = participant.sync_score or 0.0

    expected = sum([has_audio, has_video, has_transcript])
    present = sum(
        [
            bool(participant.audio_length),
            bool(participant.video_length),
            bool(participant.transcript_length),
        ]
    )
    completeness = present / expected if expected else 0.0

    overall = (audio_score + video_score + transcript_score + sync_score + completeness) / 5.0
    return {
        "overall": round(overall, 4),
        "audio": round(audio_score, 4),
        "video": round(video_score, 4),
        "transcript": round(transcript_score, 4),
        "synchronization": round(sync_score, 4),
        "metadata_completeness": round(completeness, 4),
    }


def below_threshold(scores: dict[str, float]) -> bool:
    threshold = float(dataset_config().get("quality", {}).get("min_overall_score", 0.5))
    return scores.get("overall", 0.0) < threshold
