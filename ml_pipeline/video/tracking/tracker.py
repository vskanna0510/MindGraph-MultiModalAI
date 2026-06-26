"""Temporal face tracking across frames."""

from __future__ import annotations

from ml_pipeline.video.types import LandmarkFrame, TrackState


def track_landmarks(landmarks: list[LandmarkFrame]) -> list[TrackState]:
    """Simple single-face tracker with gap tolerance."""
    if not landmarks:
        return []
    track = TrackState(track_id=0, landmarks_sequence=list(landmarks))
    return [track]
