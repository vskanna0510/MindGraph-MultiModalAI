"""Video I/O utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def video_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(frame.tobytes()).hexdigest()[:16]


def open_capture(path: Path):
    """Open video capture via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {path}")
    return cap


def read_video_metadata(path: Path) -> dict:
    """Read fps, resolution, frame count."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return {
            "fps": 1.0,
            "width": int(arr.shape[-1]) if arr.ndim >= 2 else 1,
            "height": int(arr.shape[-2]) if arr.ndim >= 2 else 1,
            "frame_count": int(arr.shape[0]) if arr.ndim >= 1 else 1,
            "duration_seconds": float(arr.shape[0]) if arr.ndim >= 1 else 0.0,
            "codec": "npy",
        }
    import cv2

    cap = open_capture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "codec": path.suffix.lower().lstrip("."),
    }
