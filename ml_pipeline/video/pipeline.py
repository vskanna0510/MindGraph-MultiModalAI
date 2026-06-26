"""Video preprocessing pipeline orchestrator."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.utils.reproducibility import SeedBundle, set_global_seeds
from ml_pipeline.video.augmentation.augment import maybe_augment_frame
from ml_pipeline.video.cache.manager import VideoCacheManager
from ml_pipeline.video.config import video_config, video_paths
from ml_pipeline.video.embeddings.extractor import extract_visual_embedding
from ml_pipeline.video.export.exporter import export_visual_artifacts
from ml_pipeline.video.face_detection.detector import detect_faces
from ml_pipeline.video.features.facial import extract_facial_features
from ml_pipeline.video.mediapipe.landmarks import extract_landmarks
from ml_pipeline.video.normalization.normalize import normalize_embedding, normalize_landmarks
from ml_pipeline.video.preprocess.validation import (
    extract_metadata,
    validate_fps,
    validate_resolution,
    verify_integrity,
)
from ml_pipeline.video.quality.assessment import assess_visual_quality
from ml_pipeline.video.quality.validation import validate_visual_pipeline
from ml_pipeline.video.sampling.frames import extract_frames, sample_frames, validate_scene
from ml_pipeline.video.tracking.tracker import track_landmarks
from ml_pipeline.video.types import PipelineStageResult, VideoPipelineResult


class VideoPipeline:
    """Production visual processing pipeline (MP2 Part 4)."""

    def __init__(self) -> None:
        self.cfg = video_config()
        seeds = SeedBundle(python=int(self.cfg.get("pipeline", {}).get("seed", 42)))
        set_global_seeds(seeds, deterministic=self.cfg.get("pipeline", {}).get("deterministic", True))
        self.cache = VideoCacheManager()
        self.paths = video_paths()

    def process_file(
        self,
        source_path: Path,
        participant_id: str,
        session_id: str | None = None,
        split: str | None = None,
    ) -> VideoPipelineResult:
        result = VideoPipelineResult(participant_id=participant_id, source_path=source_path)
        stages: list[PipelineStageResult] = []

        ok, errs = verify_integrity(source_path)
        stages.append(PipelineStageResult("integrity", ok, errors=errs))
        if not ok:
            result.stages = stages
            result.validation_passed = False
            return result

        meta = extract_metadata(source_path, participant_id, session_id)
        result.metadata = meta
        stages.append(PipelineStageResult("metadata", True))

        fps_errs = validate_fps(meta)
        res_errs = validate_resolution(meta)
        stages.append(PipelineStageResult("fps_validation", not fps_errs, errors=fps_errs))
        stages.append(PipelineStageResult("resolution_validation", not res_errs, errors=res_errs))

        frames = extract_frames(source_path, participant_id)
        frames = sample_frames(frames, meta.fps)
        if split == "train":
            try:
                import cv2

                for fr in frames:
                    if fr.path and fr.path.exists() and fr.path.suffix != ".npy":
                        img = cv2.imread(str(fr.path))
                        if img is not None:
                            cv2.imwrite(str(fr.path), maybe_augment_frame(img, split))
            except ImportError:
                pass
        result.frames = frames
        stages.append(PipelineStageResult("frame_extraction", bool(frames), metadata={"count": len(frames)}))
        stages.append(PipelineStageResult("frame_sampling", True, metadata={"count": len(frames)}))

        scene_errs = validate_scene(frames)
        stages.append(PipelineStageResult("scene_validation", not scene_errs, errors=scene_errs))

        detections = detect_faces(frames)
        result.detections = detections
        stages.append(PipelineStageResult("face_detection", True, metadata={"faces": len(detections)}))
        stages.append(PipelineStageResult("face_alignment", True))

        landmarks = extract_landmarks(frames, detections)
        result.landmarks = landmarks
        stages.append(PipelineStageResult("landmarks", bool(landmarks), metadata={"count": len(landmarks)}))

        tracks = track_landmarks(landmarks)
        result.tracks = tracks
        stages.append(PipelineStageResult("tracking", True, metadata={"tracks": len(tracks)}))

        features = extract_facial_features(landmarks)
        result.facial_features = features
        stages.append(PipelineStageResult("facial_features", True))

        assess_visual_quality(frames, detections, landmarks, meta)
        stages.append(PipelineStageResult("quality", meta.quality_score >= 0.4))

        embedding = extract_visual_embedding(frames, participant_id, meta.session_id, source_path)
        vec = np.load(embedding.vector_path)
        np.save(embedding.vector_path, normalize_embedding(vec))
        result.embedding = embedding
        stages.append(PipelineStageResult("embeddings", True))

        if landmarks:
            normed = normalize_landmarks(np.stack([lm.landmarks for lm in landmarks]))
            out = self.paths["processed"] / "landmarks" / f"{participant_id}_landmarks_norm_v1.npy"
            np.save(out, normed)
        stages.append(PipelineStageResult("normalization", True))

        valid, val_errs = validate_visual_pipeline(landmarks, features, embedding)
        result.validation_passed = valid
        stages.append(PipelineStageResult("validation", valid, errors=val_errs))

        self.cache.put(self.cache.key(embedding.video_hash, "embedding", embedding.model_version), embedding.vector_path)
        stages.append(PipelineStageResult("cache", True))

        export_visual_artifacts(result)
        stages.append(PipelineStageResult("export", True))

        result.stages = stages
        return result

    def process_batch(self, records: list[dict]) -> list[VideoPipelineResult]:
        mp_cfg = self.cfg.get("multiprocessing", {})
        if not mp_cfg.get("enabled", True) or len(records) <= 1:
            return [
                self.process_file(Path(r["file_path"]), r["participant_id"], r.get("session_id"), r.get("split"))
                for r in records
            ]
        workers = max(1, (os.cpu_count() or 1) - 1)
        results: list[VideoPipelineResult] = []
        with ProcessPoolExecutor(max_workers=min(workers, len(records))) as pool:
            futures = {
                pool.submit(_worker, r["file_path"], r["participant_id"], r.get("session_id"), r.get("split")): r
                for r in records
            }
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception:
                    pass
        return results

    def save_metadata_parquet(self, results: list[VideoPipelineResult]) -> Path:
        rows = [r.metadata.__dict__ for r in results if r.metadata]
        out = self.paths["reports"] / "video_metadata.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            pd.DataFrame(rows).to_parquet(out, index=False)
        return out


def _worker(file_path: str, participant_id: str, session_id: str | None, split: str | None):
    return VideoPipeline().process_file(Path(file_path), participant_id, session_id, split)
