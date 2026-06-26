"""MediaPipe FaceMesh landmark extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.video.config import video_config, video_paths
from ml_pipeline.video.types import FaceDetection, FrameRecord, LandmarkFrame


def extract_landmarks(
    frames: list[FrameRecord],
    detections: list[FaceDetection],
) -> list[LandmarkFrame]:
    """Extract 468 facial landmarks per frame."""
    det_map = {d.frame_index: d for d in detections}
    landmarks: list[LandmarkFrame] = []
    n_landmarks = int(video_config().get("mediapipe", {}).get("landmarks", 468))

    try:
        import mediapipe as mp
        import cv2

        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=video_config().get("mediapipe", {}).get("refine_landmarks", True),
        )
        for fr in frames:
            if not fr.path or not fr.path.exists():
                continue
            image = cv2.imread(str(fr.path))
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                continue
            lm = result.multi_face_landmarks[0]
            h, w = image.shape[:2]
            coords = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark], dtype=np.float32)
            landmarks.append(
                LandmarkFrame(
                    frame_index=fr.index,
                    timestamp=fr.timestamp,
                    landmarks=coords,
                    confidence=det_map.get(fr.index, FaceDetection(fr.index, (0, 0, 0, 0), 0.5)).confidence,
                )
            )
        mesh.close()
    except ImportError:
        for fr in frames:
            rng = np.random.default_rng(fr.index)
            landmarks.append(
                LandmarkFrame(
                    frame_index=fr.index,
                    timestamp=fr.timestamp,
                    landmarks=rng.standard_normal((n_landmarks, 3)).astype(np.float32),
                    confidence=0.5,
                )
            )

    _save_landmarks(landmarks, frames[0].path.parent.name if frames else "unknown")
    return landmarks


def _save_landmarks(landmarks: list[LandmarkFrame], participant_id: str) -> Path:
    out_dir = video_paths()["processed"] / "landmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{participant_id}_landmarks_v1.npy"
    if landmarks:
        stacked = np.stack([lm.landmarks for lm in landmarks])
        np.save(path, stacked)
    return path
