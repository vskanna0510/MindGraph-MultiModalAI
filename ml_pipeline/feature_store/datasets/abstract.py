"""Abstract dataset contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AbstractDataset(ABC):
    """Every dataset implements the full interface — no hidden side effects."""

    name: str

    @abstractmethod
    def discover(self) -> list[dict]:
        """Discover session records."""

    @abstractmethod
    def load_metadata(self) -> Any:
        """Load session metadata index."""

    @abstractmethod
    def load_features(self, record: dict) -> Any:
        """Load feature tensors for a session."""

    @abstractmethod
    def load_labels(self, record: dict) -> Any:
        """Load label for a session."""

    @abstractmethod
    def validate(self, record: dict) -> bool:
        """Validate session features."""

    def cache(self, record: dict) -> dict:
        return record

    def statistics(self) -> dict[str, Any]:
        return {}

    def export(self, target: str = "training") -> dict[str, Any]:
        return {}

    def collate(self, batch: list) -> dict[str, Any]:
        from ml_pipeline.feature_store.collate import collate_multimodal_batch

        return collate_multimodal_batch(batch)
