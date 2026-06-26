"""Frame extraction and sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_pipeline.video import config as video_config_module
from ml_pipeline.video.config import video_config
from ml_pipeline.video.types import FrameRecord
from ml_pipeline.video.utils.io import frame_hash, open_capture


def extract_frames(path: Path, participant_id: str) -> list[FrameRecord]:
    """Extract frames with metadata; cache as PNG/JPEG."""
    cfg = video_config()
    fmt = cfg.get("video", {}).get("frame_format", "png")
    out_dir = video_config_module.video_paths()["processed"] / "frames" / participant_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[FrameRecord] = []

    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        for i in range(min(len(arr), int(cfg.get("video", {}).get("max_frames", 300)))):
            frame = arr[i] if arr.ndim >= 2 else arr
            if frame.ndim == 1:
                frame = frame.reshape(1, -1)
            fpath = out_dir / f"frame_{i:05d}"
            saved_path = fpath.with_suffix(f".{fmt}" if _has_cv2() else ".npy")
            _save_frame(frame, saved_path)
            records.append(
                FrameRecord(index=i, timestamp=float(i), path=saved_path, frame_hash=frame_hash(frame.astype(np.uint8)))
            )
        return records

    import cv2

    cap = open_capture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    idx = 0
    saved = 0
    max_frames = int(cfg.get("video", {}).get("max_frames", 300))
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        fpath = out_dir / f"frame_{idx:05d}.{fmt}"
        cv2.imwrite(str(fpath), frame)
        records.append(
            FrameRecord(
                index=idx,
                timestamp=round(idx / fps, 3),
                path=fpath,
                frame_hash=frame_hash(frame),
            )
        )
        idx += 1
        saved += 1
    cap.release()
    return records


def sample_frames(frames: list[FrameRecord], fps: float) -> list[FrameRecord]:
    """Sample frames by configured strategy."""
    cfg = video_config()
    strategy = cfg.get("sampling", {}).get("strategy", "target_fps")
    target = float(cfg.get("video", {}).get("target_fps", 1))

    if strategy == "all" or not frames:
        return frames
    if strategy == "target_fps" and fps > 0:
        step = max(1, int(round(fps / target)))
        return frames[::step]
    return frames[:: max(1, len(frames) // min(len(frames), int(target * (frames[-1].timestamp or 1))))]


def _has_cv2() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def validate_scene(frames: list[FrameRecord]) -> list[str]:
    if not frames:
        return ["no_frames"]
    return []


def _save_frame(frame: np.ndarray, path: Path) -> None:
    if frame.dtype != np.uint8:
        frame = ((frame - frame.min()) / (float(np.max(frame) - np.min(frame)) + 1e-8) * 255).astype(np.uint8)
    try:
        import cv2

        if frame.ndim == 2:
            cv2.imwrite(str(path), frame)
        else:
            cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame)
    except ImportError:
        np.save(path.with_suffix(".npy"), frame)
