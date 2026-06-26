"""Base graph builder contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ml_pipeline.graph.graph_store.memory_store import InMemoryGraphStore
from ml_pipeline.graph.schema.types import GraphSnapshot


class BaseGraphBuilder(ABC):
    def __init__(self, store: InMemoryGraphStore | None = None) -> None:
        self.store = store or InMemoryGraphStore()

    @abstractmethod
    def build(self, data: dict[str, Any]) -> GraphSnapshot:
        """Build graph from session/participant data."""

    def validate(self) -> tuple[bool, list[str]]:
        from ml_pipeline.graph.evaluation.validator import GraphValidator

        snap = self.store.snapshot()
        return GraphValidator().validate(snap)
