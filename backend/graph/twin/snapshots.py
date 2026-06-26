"""Digital twin snapshot storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from graph.twin.digital_twin import DigitalTwin


class TwinSnapshotGranularity(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    EXPERIMENT = "experiment"
    VERSION = "version"


@dataclass
class TwinSnapshot:
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    granularity: str = TwinSnapshotGranularity.DAILY.value
    twin_data: dict[str, Any] = field(default_factory=dict)
    evolution_score: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    experiment_id: str | None = None
    version: str = "2.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "user_id": self.user_id,
            "granularity": self.granularity,
            "twin_data": self.twin_data,
            "evolution_score": self.evolution_score,
            "created_at": self.created_at,
            "experiment_id": self.experiment_id,
            "version": self.version,
        }


class TwinSnapshotStore:
    """Store and compare digital twin snapshots."""

    def __init__(self) -> None:
        self._snapshots: dict[str, list[TwinSnapshot]] = {}

    def capture(
        self,
        twin: DigitalTwin,
        evolution_score: dict[str, Any],
        granularity: TwinSnapshotGranularity = TwinSnapshotGranularity.DAILY,
        *,
        experiment_id: str | None = None,
    ) -> TwinSnapshot:
        snap = TwinSnapshot(
            user_id=twin.user_id,
            granularity=granularity.value,
            twin_data=twin.to_dict(),
            evolution_score=evolution_score,
            experiment_id=experiment_id,
        )
        self._snapshots.setdefault(twin.user_id, []).append(snap)
        return snap

    def list_snapshots(self, user_id: str) -> list[TwinSnapshot]:
        return list(self._snapshots.get(user_id, []))

    def compare(self, user_id: str, snap_a: str, snap_b: str) -> dict[str, Any]:
        snaps = {s.snapshot_id: s for s in self._snapshots.get(user_id, [])}
        a, b = snaps.get(snap_a), snaps.get(snap_b)
        if not a or not b:
            return {"error": "Snapshot not found"}
        score_a = a.evolution_score.get("overall_score", 0)
        score_b = b.evolution_score.get("overall_score", 0)
        return {
            "snapshot_a": snap_a,
            "snapshot_b": snap_b,
            "evolution_delta": round(score_b - score_a, 4),
            "sessions_delta": (
                b.twin_data.get("temporal_memory_sessions", 0)
                - a.twin_data.get("temporal_memory_sessions", 0)
            ),
        }
