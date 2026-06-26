"""Graph snapshot generation for rollback and audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from neo4j import AsyncSession

from graph.queries.longitudinal import user_snapshot_subgraph
from graph.repositories.base import BaseGraphRepository


class SnapshotGranularity(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    EXPERIMENT = "experiment"


@dataclass
class GraphSnapshot:
    """Point-in-time graph memory snapshot."""

    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    granularity: str = SnapshotGranularity.DAILY.value
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    node_count: int = 0
    checksum: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rollback_supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "user_id": self.user_id,
            "granularity": self.granularity,
            "created_at": self.created_at,
            "node_count": self.node_count,
            "checksum": self.checksum,
            "metadata": self.metadata,
            "rollback_supported": self.rollback_supported,
        }


class SnapshotManager:
    """Generate and store graph snapshots — daily/weekly/monthly/experiment."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = BaseGraphRepository(session)
        self._snapshots: dict[str, list[GraphSnapshot]] = {}

    @staticmethod
    def _checksum(data: Any) -> str:
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def capture(
        self,
        user_id: str,
        granularity: SnapshotGranularity = SnapshotGranularity.DAILY,
        *,
        experiment_id: str | None = None,
    ) -> GraphSnapshot:
        cypher, params = user_snapshot_subgraph(user_id)
        records, _ = await self._repo._run(cypher, params)
        node_count = int(records[0].get("node_count", 0)) if records else 0

        metadata: dict[str, Any] = {"granularity": granularity.value}
        if experiment_id:
            metadata["experiment_id"] = experiment_id

        snapshot = GraphSnapshot(
            user_id=user_id,
            granularity=granularity.value,
            node_count=node_count,
            checksum=self._checksum({"user_id": user_id, "node_count": node_count, "ts": datetime.now(UTC).isoformat()}),
            metadata=metadata,
        )

        self._snapshots.setdefault(user_id, []).append(snapshot)
        return snapshot

    def list_snapshots(self, user_id: str) -> list[GraphSnapshot]:
        return list(self._snapshots.get(user_id, []))

    def latest(self, user_id: str) -> GraphSnapshot | None:
        snaps = self._snapshots.get(user_id, [])
        return snaps[-1] if snaps else None

    def rollback_reference(self, user_id: str, snapshot_id: str) -> GraphSnapshot | None:
        for snap in self._snapshots.get(user_id, []):
            if snap.snapshot_id == snapshot_id:
                return snap
        return None
