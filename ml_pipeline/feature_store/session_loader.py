"""Session loader — resolves features into tensors."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ml_pipeline.feature_store.config import feature_store_config, feature_store_paths
from ml_pipeline.feature_store.store import FeatureStore
from ml_pipeline.feature_store.types import FeatureType, MultimodalSession, SessionBatch


class SessionLoader:
    def __init__(self) -> None:
        self.cfg = feature_store_config()
        self.paths = feature_store_paths()
        self.store = FeatureStore(self.paths["root"])
        self.versions = self.cfg.get("feature_store", {}).get("feature_versions", {})
        self.extraction = self.cfg.get("feature_store", {}).get("extraction_versions", {})
        self.models = self.cfg.get("feature_store", {}).get("model_versions", {})

    def _find_feature_path(self, modality: str, participant_id: str, session_id: str) -> Path | None:
        candidates: list[Path] = []
        pid = participant_id
        if modality == "audio":
            roots = [self.paths["source_audio"], self.paths["audio"]]
            patterns = [f"{pid}_audio_v1.npy", f"{pid}_preprocessed_v1.wav", f"{pid}_AUDIO_features.npy"]
        elif modality == "visual":
            roots = [self.paths["source_visual"], self.paths["visual"]]
            patterns = [f"{pid}_visual_v1.npy", f"{pid}_{session_id}_visual.npy", f"{pid}_landmarks_norm_v1.npy"]
        elif modality == "text":
            roots = [self.paths["source_text"] / "embeddings", self.paths["text"]]
            patterns = [f"{pid}_{session_id}_text.npy", f"{pid}_text_v1.npy"]
        else:
            return None
        for root in roots:
            if not root.exists():
                continue
            for pat in patterns:
                p = root / pat
                if p.exists():
                    return p
            for p in root.glob(f"{pid}*"):
                if p.suffix in {".npy", ".npz"}:
                    candidates.append(p)
        return candidates[0] if candidates else None

    def _load_or_synthesize(self, path: Path | None, dim: int = 128) -> np.ndarray | None:
        if path and path.exists():
            if path.suffix == ".npy":
                return self.store.load_numpy(path)
            # Feature store uses processed embeddings only — never load full raw WAV here.
        return None

    def load_session(self, session: MultimodalSession) -> SessionBatch:
        start = time.perf_counter()
        meta = session.metadata
        audio_path = Path(meta.get("audio_path", "")) if meta.get("audio_path") else self._find_feature_path("audio", session.participant_id, session.session_id)
        visual_path = Path(meta.get("video_path", "")) if meta.get("video_path") else self._find_feature_path("visual", session.participant_id, session.session_id)
        text_path = Path(meta.get("transcript_path", "")) if meta.get("transcript_path") else self._find_feature_path("text", session.participant_id, session.session_id)

        audio = self._load_or_synthesize(audio_path) if session.modality_mask.get("audio") else None
        visual = self._load_or_synthesize(visual_path, 768) if session.modality_mask.get("visual") else None
        text = self._load_or_synthesize(text_path, 768) if session.modality_mask.get("text") else None

        label = session.label
        label_int = 1 if str(label).lower() in {"depression", "1", "depressed"} else 0 if str(label).lower() in {"normal", "0"} else -1

        mask = {
            "audio": audio is not None,
            "visual": visual is not None,
            "text": text is not None,
            "image": False,
            "graph": False,
        }

        graph_data = None
        try:
            from ml_pipeline.graph.pipeline import GraphPipeline

            gp = GraphPipeline()
            session_payload = {
                "participant_id": session.participant_id,
                "session_id": session.session_id,
                "dataset": session.dataset,
                "language": session.language,
                "label": session.label,
                "quality_score": session.quality.overall,
                "modalities": session.modality_mask,
                "timestamp": session.timestamp,
            }
            tensors = gp.build_tensors(session_payload)
            graph_data = tensors.kg_vector.numpy()
            mask["graph"] = True
        except Exception:
            graph_data = None

        return SessionBatch(
            participant_id=session.participant_id,
            session_id=session.session_id,
            audio=audio,
            visual=visual,
            text=text,
            image=None,
            graph=graph_data,
            clinical=session.clinical,
            label=label_int,
            quality=session.quality,
            language=session.language,
            split=session.split,
            timestamp=datetime.now(UTC).isoformat(),
            modality_mask=mask,
        )
