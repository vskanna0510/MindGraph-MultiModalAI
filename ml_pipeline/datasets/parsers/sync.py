"""Audio/video synchronization and alignment scoring."""

from __future__ import annotations

from ml_pipeline.datasets.parsers.types import ParticipantMetadata, TranscriptData


def compute_sync(
    participant: ParticipantMetadata,
    transcript: TranscriptData | None,
) -> tuple[float | None, float]:
    """Compute offset and alignment score between modalities."""
    durations: list[float] = []
    if participant.audio_length:
        durations.append(participant.audio_length)
    if participant.video_length:
        durations.append(participant.video_length)
    if transcript and transcript.utterances:
        max_stop = max(u.stop_time for u in transcript.utterances)
        durations.append(max_stop)

    if len(durations) < 2:
        return None, 1.0 if durations else 0.0

    max_d = max(durations)
    min_d = min(durations)
    offset = round(max_d - min_d, 3)
    ratio = min_d / max_d if max_d > 0 else 0.0
    score = round(min(1.0, ratio), 4)
    if offset > 30.0:
        score = round(score * 0.5, 4)
    return offset, score
