"""Base dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ml_pipeline.datasets.types import SampleRecord


class BaseDatasetAdapter(ABC):
    """Adapter contract — one implementation per dataset."""

    name: str
    version: str

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = raw_root

    @abstractmethod
    def discover(self) -> list[SampleRecord]:
        """Discover all samples in the raw dataset directory."""

    @abstractmethod
    def expected_structure(self) -> list[str]:
        """Return relative paths or patterns expected in raw_root."""

    def participant_ids(self) -> list[str]:
        """Unique participant IDs from discovered samples."""
        return sorted({record.participant_id for record in self.discover()})
