"""Feature storage with lazy loading and checksums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline.feature_store.types import FeatureRecord, FeatureType


def checksum_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def checksum_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


class FeatureStore:
    """Read/write traceable feature artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_numpy(
        self,
        array: np.ndarray,
        record: FeatureRecord,
    ) -> Path:
        out_dir = self.root / record.feature_type.value
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{record.feature_id}.npy"
        np.save(path, array)
        meta_path = out_dir / f"{record.feature_id}.json"
        meta_path.write_text(
            json.dumps(
                {
                    "feature_id": record.feature_id,
                    "participant_id": record.participant_id,
                    "session_id": record.session_id,
                    "feature_type": record.feature_type.value,
                    "feature_version": record.feature_version,
                    "extraction_version": record.extraction_version,
                    "model_version": record.model_version,
                    "checksum": checksum_array(array),
                    "embedding_shape": list(record.embedding_shape),
                    "embedding_dim": record.embedding_dim,
                    "quality_score": record.quality_score,
                    "storage_path": str(path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def load_numpy(self, path: Path, mmap: bool = True) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(path)
        if mmap:
            return np.load(path, mmap_mode="r")
        return np.load(path)

    def load_lazy(self, record: FeatureRecord) -> np.ndarray:
        return self.load_numpy(record.storage_path, mmap=True)

    def resolve_or_create_placeholder(
        self,
        path: Path | None,
        participant_id: str,
        session_id: str,
        feature_type: FeatureType,
        feature_version: str,
        extraction_version: str,
        model_version: str,
        dataset: str,
        language: str,
        split: str,
        dim: int = 128,
    ) -> tuple[FeatureRecord | None, np.ndarray | None]:
        if path and path.exists():
            arr = self.load_numpy(path)
            shape = tuple(arr.shape)
            dim = int(arr.shape[-1]) if arr.ndim > 1 else int(arr.shape[0])
            record = FeatureRecord(
                feature_id=f"{participant_id}_{session_id}_{feature_type.value}_{feature_version}",
                participant_id=participant_id,
                session_id=session_id,
                feature_type=feature_type,
                feature_version=feature_version,
                extraction_version=extraction_version,
                model_version=model_version,
                dataset=dataset,
                language=language,
                split=split,
                embedding_shape=shape,
                embedding_dim=dim,
                normalization_method="none",
                storage_path=path,
                checksum=checksum_file(path),
                quality_score=0.8,
                extraction_time="",
                processing_duration_ms=0.0,
                cache_status="hit",
            )
            return record, arr
        return None, None
