"""Base multimodal PyTorch dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_pipeline.feature_store.config import feature_store_config
from ml_pipeline.feature_store.datasets.abstract import AbstractDataset
from ml_pipeline.feature_store.index import build_session_index, load_dataset_index, sessions_to_multimodal
from ml_pipeline.feature_store.normalization import NormalizationState, apply_normalization
from ml_pipeline.feature_store.quality import assess_session_quality
from ml_pipeline.feature_store.session_loader import SessionLoader
from ml_pipeline.feature_store.statistics import compute_dataset_statistics
from ml_pipeline.feature_store.validation import validate_session_batch


class BaseMultimodalDataset(AbstractDataset):
    """Shared responsibilities: metadata, features, validation, cache, stats, export."""

    def __init__(self, split: str | None = None, augment: bool = False) -> None:
        self.cfg = feature_store_config()
        self.split = split
        self.augment = augment
        self.loader = SessionLoader()
        self.norm_state = NormalizationState()
        self._sessions = []
        self._index_df = pd.DataFrame()

    def load_metadata(self) -> pd.DataFrame:
        self._index_df = build_session_index(load_dataset_index())
        if self.split:
            self._index_df = self._index_df[self._index_df["split"] == self.split]
        self._sessions = sessions_to_multimodal(self._index_df)
        return self._index_df

    def discover(self) -> list[dict]:
        if self._index_df.empty:
            self.load_metadata()
        return self._index_df.to_dict(orient="records")

    def load_features(self, record: dict) -> Any:
        session = next(
            (s for s in self._sessions if s.participant_id == str(record["participant_id"])),
            None,
        )
        if session is None:
            from ml_pipeline.feature_store.types import MultimodalSession, QualityMetrics

            session = MultimodalSession(
                participant_id=str(record["participant_id"]),
                session_id=str(record.get("session_id", record["participant_id"])),
                dataset=str(record.get("dataset", self.name)),
                split=str(record.get("split", "unknown")),
                language=str(record.get("language", "en")),
                label=str(record.get("label", "unknown")),
                modality_mask={
                    "audio": bool(record.get("has_audio")),
                    "visual": bool(record.get("has_visual")),
                    "text": bool(record.get("has_transcript")),
                    "image": False,
                    "graph": False,
                },
                quality=QualityMetrics(overall=float(record.get("quality_score", 0.8))),
                metadata={
                    "audio_path": record.get("audio_path", ""),
                    "transcript_path": record.get("transcript_path", ""),
                    "video_path": record.get("video_path", ""),
                },
            )
        batch = self.loader.load_session(session)
        if self.norm_state.audio:
            batch = type(batch)(
                **{
                    **batch.__dict__,
                    "audio": apply_normalization(batch.audio, self.norm_state.audio),
                    "visual": apply_normalization(batch.visual, self.norm_state.visual),
                    "text": apply_normalization(batch.text, self.norm_state.text),
                }
            )
        return batch

    def load_labels(self, record: dict) -> Any:
        label = str(record.get("label", "unknown"))
        if label.lower() in {"depression", "depressed", "1"}:
            return 1
        if label.lower() in {"normal", "0"}:
            return 0
        return -1

    def validate(self, record: dict) -> bool:
        batch = self.load_features(record)
        ok, _ = validate_session_batch(batch)
        return ok

    def statistics(self) -> dict:
        return compute_dataset_statistics(self._index_df)

    def __len__(self) -> int:
        if self._index_df.empty:
            self.load_metadata()
        return len(self._index_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.discover()[idx]
        batch = self.load_features(record)
        return {
            "participant_id": batch.participant_id,
            "session_id": batch.session_id,
            "audio": batch.audio,
            "visual": batch.visual,
            "text": batch.text,
            "label": batch.label,
            "modality_mask": batch.modality_mask,
            "split": batch.split,
            "quality": batch.quality.overall,
        }
