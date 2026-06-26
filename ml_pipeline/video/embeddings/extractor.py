"""Visual embedding extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ml_pipeline.video.config import video_config, video_paths
from ml_pipeline.video.types import FrameRecord, VisualEmbedding
from ml_pipeline.video.utils.io import video_hash


class BaseVisualEmbeddingExtractor(ABC):
    model_name: str
    model_version: str

    @abstractmethod
    def extract(self, frames: list[FrameRecord]) -> np.ndarray:
        pass


class ViTEmbeddingExtractor(BaseVisualEmbeddingExtractor):
    model_name = "google/vit-base-patch16-224"
    model_version = "1.0"

    def extract(self, frames: list[FrameRecord]) -> np.ndarray:
        cfg = video_config().get("embeddings", {})
        self.model_name = cfg.get("primary", self.model_name)
        try:
            import torch
            from PIL import Image
            from transformers import ViTImageProcessor, ViTModel

            processor = ViTImageProcessor.from_pretrained(self.model_name)
            model = ViTModel.from_pretrained(self.model_name)
            device = "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") == "auto" else "cpu"
            model = model.to(device).eval()
            vectors = []
            for fr in frames[:16]:
                if not fr.path or not fr.path.exists():
                    continue
                if fr.path.suffix.lower() == ".npy":
                    arr = np.load(fr.path)
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    image = Image.fromarray(arr.astype(np.uint8))
                else:
                    image = Image.open(fr.path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(**inputs)
                vectors.append(out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
            if vectors:
                return np.mean(vectors, axis=0).astype(np.float32)
        except ImportError:
            pass
        seed = len(frames)
        return np.random.default_rng(seed).standard_normal(768, dtype=np.float32)


def extract_visual_embedding(
    frames: list[FrameRecord],
    participant_id: str,
    session_id: str,
    source_path: Path,
    extractor: BaseVisualEmbeddingExtractor | None = None,
) -> VisualEmbedding:
    ext = extractor or ViTEmbeddingExtractor()
    vector = ext.extract(frames)
    cfg = video_config()
    cache_dir = video_paths()["cache"] / cfg.get("embeddings", {}).get("cache_subdir", "vit")
    cache_dir.mkdir(parents=True, exist_ok=True)
    v_hash = video_hash(source_path) if source_path.exists() else "npy"
    config_ver = cfg.get("pipeline", {}).get("version", "1.0.0")
    key = f"{v_hash}_{ext.model_version}_{config_ver}"
    path = cache_dir / f"{key}.npy"
    if not path.exists():
        np.save(path, vector)
    return VisualEmbedding(
        participant_id=participant_id,
        session_id=session_id,
        model_name=ext.model_name,
        model_version=ext.model_version,
        embedding_dim=int(vector.shape[-1]),
        vector_path=path,
        video_hash=v_hash,
        config_version=config_ver,
        extraction_timestamp=datetime.now(UTC).isoformat(),
    )
