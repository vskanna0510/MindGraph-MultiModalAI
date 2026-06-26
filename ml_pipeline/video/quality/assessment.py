"""Visual quality assessment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.video.types import FaceDetection, FrameRecord, LandmarkFrame, VideoMetadata


def assess_visual_quality(
    frames: list[FrameRecord],
    detections: list[FaceDetection],
    landmarks: list[LandmarkFrame],
    metadata: VideoMetadata,
) -> tuple[float, dict]:
    brightness_vals, blur_vals = [], []
    try:
        import cv2

        for fr in frames[:20]:
            if not fr.path or not fr.path.exists():
                continue
            if fr.path.suffix == ".npy":
                img = np.load(fr.path)
                if img.ndim == 3:
                    img = np.mean(img, axis=-1)
            else:
                img = cv2.imread(str(fr.path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            brightness_vals.append(float(np.mean(img)))
            blur_vals.append(float(cv2.Laplacian(img.astype(np.float32), cv2.CV_64F).var()))
    except ImportError:
        brightness_vals = [128.0]
        blur_vals = [100.0]

    face_rate = len(detections) / max(len(frames), 1)
    lm_rate = len(landmarks) / max(len(frames), 1)
    metadata.face_detection_rate = round(face_rate, 4)
    metadata.landmark_success_rate = round(lm_rate, 4)

    blur_score = 1.0 - min(1.0, float(np.mean(blur_vals)) / 500.0) if blur_vals else 0.5
    score = 0.3 * face_rate + 0.3 * lm_rate + 0.2 * (1 - blur_score) + 0.2 * min(1.0, metadata.duration_seconds / 60)
    metadata.quality_score = round(min(1.0, score), 4)
    return metadata.quality_score, {
        "brightness": round(float(np.mean(brightness_vals)), 2) if brightness_vals else 0,
        "blur": round(float(np.mean(blur_vals)), 2) if blur_vals else 0,
        "face_rate": face_rate,
    }
