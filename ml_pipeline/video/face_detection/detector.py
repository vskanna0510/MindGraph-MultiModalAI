"""Face detection — MediaPipe primary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.video.config import video_config
from ml_pipeline.video.types import FaceDetection, FrameRecord


def detect_faces(frames: list[FrameRecord]) -> list[FaceDetection]:
    """Run face detection on sampled frames."""
    cfg = video_config()
    method = cfg.get("face_detection", {}).get("primary", "mediapipe")
    min_conf = float(cfg.get("face_detection", {}).get("min_confidence", 0.6))
    detections: list[FaceDetection] = []

    if method == "mediapipe":
        try:
            import cv2
            import mediapipe as mp

            for fr in frames:
                if not fr.path or not fr.path.exists():
                    continue
                image = cv2.imread(str(fr.path))
                if image is None:
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = mp_face.process(rgb)
                if result.detections:
                    det = result.detections[0]
                    bb = det.location_data.relative_bounding_box
                    h, w = image.shape[:2]
                    detections.append(
                        FaceDetection(
                            frame_index=fr.index,
                            bbox=(bb.xmin * w, bb.ymin * h, bb.width * w, bb.height * h),
                            confidence=float(det.score[0]),
                            face_count=len(result.detections),
                        )
                    )
            mp_face.close()
            return detections
        except ImportError:
            pass

    # Fallback: center crop heuristic
    for fr in frames:
        detections.append(
            FaceDetection(frame_index=fr.index, bbox=(0.25, 0.25, 0.5, 0.5), confidence=0.5)
        )
    return detections
