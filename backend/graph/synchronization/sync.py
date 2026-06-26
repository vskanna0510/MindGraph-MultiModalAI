"""Offline/edge graph synchronization stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SyncBatch:
    batch_id: str
    operations: list[dict[str, Any]] = field(default_factory=list)
    synced: bool = False


class GraphSynchronizer:
    """Queue offline mutations for later sync to Neo4j."""

    def __init__(self) -> None:
        self._pending: list[SyncBatch] = []

    def enqueue(self, batch: SyncBatch) -> None:
        self._pending.append(batch)

    def pending_count(self) -> int:
        return len(self._pending)

    def mark_synced(self, batch_id: str) -> bool:
        for batch in self._pending:
            if batch.batch_id == batch_id:
                batch.synced = True
                return True
        return False
