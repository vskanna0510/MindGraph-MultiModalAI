"""Facial feature extraction from landmarks."""

from __future__ import annotations

import numpy as np

from ml_pipeline.video.types import FacialFeatures, LandmarkFrame


def _eye_aspect_ratio(landmarks: np.ndarray) -> float:
    if landmarks.shape[0] < 400:
        return 0.3
    left = landmarks[33:42]
    right = landmarks[263:272]
    def ear(pts):
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3]) + 1e-8
        return (v1 + v2) / (2.0 * h)
    return float((ear(left) + ear(right)) / 2.0)


def _mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    if landmarks.shape[0] < 300:
        return 0.2
    top, bottom = landmarks[13], landmarks[14]
    left, right = landmarks[61], landmarks[291]
    return float(np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-8))


def extract_facial_features(landmarks: list[LandmarkFrame]) -> FacialFeatures:
    ears, mars = [], []
    yaws, pitches, rolls = [], [], []
    for lm in landmarks:
        pts = lm.landmarks
        ears.append(_eye_aspect_ratio(pts))
        mars.append(_mouth_aspect_ratio(pts))
        nose, chin = pts[1], pts[152] if pts.shape[0] > 152 else pts[-1]
        yaws.append(float(nose[0] - chin[0]))
        pitches.append(float(nose[1] - chin[1]))
        rolls.append(float(nose[2] - chin[2]))

    ear_arr = np.array(ears, dtype=np.float32)
    blink_thresh = 0.2
    blinks = int(np.sum(ear_arr < blink_thresh))
    duration = max(landmarks[-1].timestamp, 1.0) if landmarks else 1.0

    left = landmarks[0].landmarks[:, 0] if landmarks else np.array([0])
    right = landmarks[0].landmarks[:, 0] if landmarks else np.array([0])
    symmetry = 1.0 - float(np.mean(np.abs(left - right[::-1][: len(left)]))) / (np.std(left) + 1e-8)

    return FacialFeatures(
        ear=ear_arr,
        mar=np.array(mars, dtype=np.float32),
        blink_rate=round(blinks / duration, 4),
        smile_intensity=round(float(np.mean(mars)), 4),
        head_pose_yaw=np.array(yaws, dtype=np.float32),
        head_pose_pitch=np.array(pitches, dtype=np.float32),
        head_pose_roll=np.array(rolls, dtype=np.float32),
        symmetry_score=round(max(0.0, min(1.0, symmetry)), 4),
        gaze_forward_ratio=round(float(np.mean(ear_arr > blink_thresh)), 4),
    )
