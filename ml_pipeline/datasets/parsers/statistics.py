"""Parser statistics generation."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from ml_pipeline.datasets.parsers.types import ParticipantMetadata, ParserResult


def compute_statistics(result: ParserResult) -> dict[str, Any]:
    """Generate dataset statistics from parser result."""
    participants = result.participants
    durations = [p.interview_duration for p in participants if p.interview_duration]
    audio_lengths = [p.audio_length for p in participants if p.audio_length]
    transcript_lengths = [p.transcript_length for p in participants if p.transcript_length]
    video_lengths = [p.video_length for p in participants if p.video_length]

    return {
        "participant_count": len(participants),
        "session_count": len({p.session_id for p in participants}),
        "average_duration": round(statistics.mean(durations), 2) if durations else 0,
        "median_duration": round(statistics.median(durations), 2) if durations else 0,
        "total_hours": round(sum(durations) / 3600, 2) if durations else 0,
        "label_distribution": dict(Counter(p.label.value for p in participants)),
        "gender_distribution": dict(Counter(p.gender for p in participants if p.gender)),
        "split_distribution": dict(Counter(p.split for p in participants if p.split)),
        "language_distribution": dict(Counter(p.language.value for p in participants)),
        "avg_audio_length": round(statistics.mean(audio_lengths), 2) if audio_lengths else 0,
        "avg_transcript_length": round(statistics.mean(transcript_lengths), 2) if transcript_lengths else 0,
        "avg_video_length": round(statistics.mean(video_lengths), 2) if video_lengths else 0,
        "validation_issue_count": len(result.validation_issues),
    }
