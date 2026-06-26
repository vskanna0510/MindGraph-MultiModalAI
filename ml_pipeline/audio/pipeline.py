"""Audio engineering pipeline orchestrator — 16 stages in fixed order."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pipeline.audio.augmentation.augment import maybe_augment
from ml_pipeline.audio.cache.manager import AudioCacheManager
from ml_pipeline.audio.config import audio_config, audio_paths
from ml_pipeline.audio.embeddings.extractor import extract_embedding
from ml_pipeline.audio.export.exporter import export_artifacts
from ml_pipeline.audio.features.extractor import extract_features
from ml_pipeline.audio.normalization.feature_norm import normalize_features
from ml_pipeline.audio.preprocess.amplitude import normalize_amplitude
from ml_pipeline.audio.preprocess.channels import to_mono
from ml_pipeline.audio.preprocess.integrity import verify_integrity
from ml_pipeline.audio.preprocess.metadata import extract_metadata
from ml_pipeline.audio.preprocess.noise import reduce_noise
from ml_pipeline.audio.preprocess.resample import resample
from ml_pipeline.audio.quality.assessment import assess_quality
from ml_pipeline.audio.quality.validation import validate_features, write_validation_report
from ml_pipeline.audio.segmentation.speech_segments import segment_speech
from ml_pipeline.audio.segmentation.vad import compute_pauses, run_vad
from ml_pipeline.audio.types import AudioPipelineResult, PipelineStageResult
from ml_pipeline.audio.utils.io import load_audio, save_wav
from ml_pipeline.utils.reproducibility import SeedBundle, set_global_seeds


class AudioPipeline:
    """Production speech processing pipeline (MP2 Part 3)."""

    def __init__(self) -> None:
        self.cfg = audio_config()
        seeds = SeedBundle(python=int(self.cfg.get("pipeline", {}).get("seed", 42)))
        set_global_seeds(seeds, deterministic=self.cfg.get("pipeline", {}).get("deterministic", True))
        self.cache = AudioCacheManager()
        self.paths = audio_paths()

    def process_file(
        self,
        source_path: Path,
        participant_id: str,
        session_id: str | None = None,
        split: str | None = None,
    ) -> AudioPipelineResult:
        """Run all 16 stages on a single audio file."""
        result = AudioPipelineResult(participant_id=participant_id, source_path=source_path)
        stages: list[PipelineStageResult] = []

        # 1 Integrity
        ok, errs = verify_integrity(source_path)
        stages.append(PipelineStageResult("integrity", ok, errors=errs))
        if not ok:
            result.stages = stages
            result.validation_passed = False
            return result

        # 2 Metadata
        meta = extract_metadata(source_path, participant_id, session_id)
        result.metadata = meta
        stages.append(PipelineStageResult("metadata", True, metadata={"duration": meta.duration_seconds}))

        # Load audio (D-VLOG npy shortcut)
        if source_path.suffix.lower() == ".npy":
            audio = np.load(source_path).astype(np.float32).ravel()
            sr = int(self.cfg.get("audio", {}).get("target_sample_rate", 16000))
        else:
            audio, sr = load_audio(source_path)

        # 3 Quality
        q_score, q_flags = assess_quality(audio, meta)
        stages.append(PipelineStageResult("quality", q_score >= 0.4, metadata={"score": q_score}, errors=q_flags))

        # 4 Resample
        audio, sr = resample(audio, sr)
        stages.append(PipelineStageResult("resample", True, metadata={"sample_rate": sr}))

        # 5 Channels
        audio = to_mono(audio, meta.original_channels)
        stages.append(PipelineStageResult("channels", True))

        # 6 Amplitude
        audio = normalize_amplitude(audio)
        stages.append(PipelineStageResult("amplitude", True))

        # 7 Noise reduction
        audio, noise_meta = reduce_noise(audio, sr)
        meta.snr_processed = noise_meta.get("snr_processed")
        stages.append(PipelineStageResult("noise_reduction", True, metadata=noise_meta))

        # 8-9 VAD + silence
        vad = run_vad(audio, sr)
        pauses = compute_pauses(audio, sr, vad)
        meta.speech_ratio = vad.speech_ratio
        result.vad = vad
        result.pauses = pauses
        stages.append(PipelineStageResult("vad", True, metadata={"speech_ratio": vad.speech_ratio}))
        stages.append(PipelineStageResult("silence", True, metadata={"silence_pct": pauses.silence_percentage}))

        # 10 Segmentation
        segments = segment_speech(audio, sr, vad)
        stages.append(PipelineStageResult("segmentation", True, metadata={"segments": len(segments)}))

        # Training-only augmentation
        audio = maybe_augment(audio, sr, split)

        # Cache processed wav
        proc_dir = self.paths["processed"] / "preprocessed"
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_path = proc_dir / f"{participant_id}_preprocessed_v1.wav"
        if not proc_path.exists():
            save_wav(proc_path, audio, sr)
        result.processed_path = proc_path

        # 11 Features
        features = extract_features(audio, sr, participant_id)
        if features.mfcc is not None:
            features.mfcc, _ = normalize_features(features.mfcc)
        result.features = features
        stages.append(PipelineStageResult("features", True))

        # 12 Embeddings
        embedding = extract_embedding(audio, sr, participant_id, meta.session_id, source_path)
        self.cache.put(embedding.checksum, embedding.vector_path)
        result.embedding = embedding
        stages.append(PipelineStageResult("embeddings", True))

        # 13 Normalization (feature-level done above)
        stages.append(PipelineStageResult("normalization", True))

        # 14 Validation
        valid, val_errs = validate_features(features, embedding)
        result.validation_passed = valid
        stages.append(PipelineStageResult("validation", valid, errors=val_errs))

        # 15 Cache (embedding cached in extract)
        stages.append(PipelineStageResult("cache", True, metadata={"hit_rate": self.cache.hit_rate}))

        # 16 Export
        export_artifacts(result)
        stages.append(PipelineStageResult("export", True))

        result.stages = stages
        return result

    def process_batch(self, records: list[dict], max_workers: int | None = None) -> list[AudioPipelineResult]:
        """Process multiple files with multiprocessing."""
        mp_cfg = self.cfg.get("multiprocessing", {})
        if not mp_cfg.get("enabled", True) or len(records) <= 1:
            return [self.process_file(Path(r["file_path"]), r["participant_id"], r.get("session_id"), r.get("split")) for r in records]

        workers = max_workers or int(mp_cfg.get("max_workers", 0))
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 1) - 1)
        workers = min(workers, len(records))
        results: list[AudioPipelineResult] = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_file_worker,
                    r["file_path"],
                    r["participant_id"],
                    r.get("session_id"),
                    r.get("split"),
                ): r
                for r in records
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass
        return results

    def save_metadata_parquet(self, results: list[AudioPipelineResult]) -> Path:
        rows = []
        for r in results:
            if r.metadata:
                rows.append(r.metadata.__dict__)
        out = self.paths["reports"] / "audio_metadata.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        if rows:
            pd.DataFrame(rows).to_parquet(out, index=False)
        return out


def _process_file_worker(file_path: str, participant_id: str, session_id: str | None, split: str | None):
  # Module-level for pickling on Windows
    return AudioPipeline().process_file(Path(file_path), participant_id, session_id, split)
