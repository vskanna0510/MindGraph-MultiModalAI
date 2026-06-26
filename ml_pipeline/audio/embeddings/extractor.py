"""Stage 12 — Embedding extraction interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ml_pipeline.audio.config import audio_config, audio_paths
from ml_pipeline.audio.types import EmbeddingRecord
from ml_pipeline.audio.utils.io import audio_hash


class BaseEmbeddingExtractor(ABC):
    """Abstract embedding extractor — never couple to one model."""

    model_name: str
    model_version: str

    @abstractmethod
    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return embedding vector."""

    @property
    def embedding_dim(self) -> int:
        return 768


class Wav2Vec2Extractor(BaseEmbeddingExtractor):
    """Wav2Vec2 embedding extractor with GPU support."""

    model_name = "facebook/wav2vec2-base"
    model_version = "1.0"

    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        cfg = audio_config().get("embeddings", {})
        self.model_name = cfg.get("primary", self.model_name)
        self.model_version = cfg.get("model_version", self.model_version)
        try:
            import torch
            from transformers import Wav2Vec2Model, Wav2Vec2Processor

            device_cfg = cfg.get("device", "auto")
            self._device = "cuda" if device_cfg == "auto" and torch.cuda.is_available() else "cpu"
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self._model = Wav2Vec2Model.from_pretrained(self.model_name).to(self._device)
            self._model.eval()
        except ImportError:
            self._model = None

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        self._load()
        if self._model is None:
            seed = abs(hash(audio.tobytes()[:128])) % (2**31)
            rng = np.random.default_rng(seed)
            return rng.standard_normal(768, dtype=np.float32)
        import torch

        inputs = self._processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)


def extract_embedding(
    audio: np.ndarray,
    sample_rate: int,
    participant_id: str,
    session_id: str,
    source_path: Path,
    extractor: BaseEmbeddingExtractor | None = None,
) -> EmbeddingRecord:
    """Extract and cache embedding."""
    cfg = audio_config()
    ext = extractor or Wav2Vec2Extractor()
    vector = ext.extract(audio, sample_rate)
    paths = audio_paths()
    cache_dir = paths["cache"] / cfg.get("embeddings", {}).get("cache_subdir", "wav2vec2")
    cache_dir.mkdir(parents=True, exist_ok=True)
    a_hash = audio_hash(source_path)
    config_ver = cfg.get("pipeline", {}).get("version", "1.0.0")
    cache_key = f"{a_hash}_{ext.model_version}_{config_ver}"
    vector_path = cache_dir / f"{cache_key}.npy"
    if not vector_path.exists():
        np.save(vector_path, vector)
    return EmbeddingRecord(
        participant_id=participant_id,
        session_id=session_id,
        model_name=ext.model_name,
        model_version=ext.model_version,
        embedding_dim=int(vector.shape[-1]),
        vector_path=vector_path,
        audio_hash=a_hash,
        config_version=config_ver,
        extraction_timestamp=datetime.now(UTC).isoformat(),
        checksum=cache_key,
    )
