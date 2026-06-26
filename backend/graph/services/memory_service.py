"""Temporal memory service facade."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncSession

from graph.retention import DataRetentionPolicy
from graph.temporal.memory_engine import MemoryUpdateResult, TemporalMemoryEngine
from graph.temporal.snapshots import SnapshotGranularity, SnapshotManager


class MemoryService:
    """High-level API for longitudinal memory operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._engine = TemporalMemoryEngine(session)
        self._snapshots = SnapshotManager(session)
        self._retention = DataRetentionPolicy(session)

    async def record_session_memory(
        self,
        user_id: str,
        session_id: str,
        *,
        previous_session_id: str | None = None,
        risk_probability: float,
        confidence: float,
        **kwargs: Any,
    ) -> MemoryUpdateResult:
        await self._engine.ensure_user(user_id)
        await self._engine.record_session(user_id, session_id, previous_session_id=previous_session_id, **kwargs)
        return await self._engine.append_inference(
            user_id,
            session_id,
            risk_probability=risk_probability,
            confidence=confidence,
            emotions=kwargs.get("emotions"),
            symptoms=kwargs.get("symptoms"),
            behaviours=kwargs.get("behaviours"),
            recommendations=kwargs.get("recommendations"),
            binary_prediction=kwargs.get("binary_prediction", 0),
            model_version=kwargs.get("model_version", "MODEL_010"),
        )

    async def snapshot(
        self,
        user_id: str,
        granularity: SnapshotGranularity = SnapshotGranularity.DAILY,
        *,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        snap = await self._snapshots.capture(user_id, granularity, experiment_id=experiment_id)
        return snap.to_dict()

    async def soft_delete_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        result = await self._retention.soft_delete_session(user_id, session_id)
        return {"action": result.action, "affected": result.affected, "timestamp": result.timestamp}

    async def gdpr_erase(self, user_id: str) -> dict[str, Any]:
        result = await self._retention.hard_delete_user(user_id)
        return {"action": result.action, "user_id": user_id, "timestamp": result.timestamp}
