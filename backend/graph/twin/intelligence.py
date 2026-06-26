"""Graph intelligence engine — orchestrates twin, insights, and forecasts."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from neo4j import AsyncSession

from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.contracts import GraphServiceResponse, TimedOperation
from graph.queries import session_queries as sq
from graph.queries import trend_queries as tq
from graph.queries import recommendation_queries as rq
from graph.repositories.base import BaseGraphRepository
from graph.security import GraphSecurityPolicy
from graph.twin.attention import AttentionVisualizer
from graph.twin.digital_twin import DigitalTwin, DigitalTwinBuilder
from graph.twin.evolution_score import EvolutionScoreCalculator
from graph.twin.risk_evolution import RiskEvolutionModel
from graph.twin.snapshots import TwinSnapshotGranularity, TwinSnapshotStore
from graph.twin.story import GraphStoryGenerator


class GraphIntelligenceEngine:
    """Primary research novelty — evolving Digital Cognitive Twin."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._repo = BaseGraphRepository(session)
        self._cache = cache or GraphCache()
        self._config = get_graph_config()
        self._twins: dict[str, DigitalTwin] = {}
        self._snapshots = TwinSnapshotStore()
        self._risk_model = RiskEvolutionModel()

    async def update_twin(
        self,
        user_id: str,
        *,
        requester_id: str,
        correlation_id: str | None = None,
    ) -> GraphServiceResponse[dict[str, Any]]:
        if not GraphSecurityPolicy.can_read(requester_id, user_id):
            raise PermissionError("Unauthorized twin access")

        cid = correlation_id or str(uuid.uuid4())
        cache_key = f"twin:{user_id}"
        cached = self._cache.get(cache_key)

        with TimedOperation() as timer:
            data = await self._fetch_user_data(user_id)
            existing = self._twins.get(user_id)
            twin = DigitalTwinBuilder.build(
                user_id,
                risk_scores=data["risk_scores"],
                emotions=data["emotions"],
                symptoms=data["symptoms"],
                behaviours=data["behaviours"],
                sessions=data["sessions"],
                recommendations=data["recommendations"],
                user_props=data.get("user_props"),
                existing=existing,
            )
            self._twins[user_id] = twin
            risk_evo = self._risk_model.analyze(twin.risk_scores)
            evolution = EvolutionScoreCalculator.compute(twin)
            self._cache.set(cache_key, twin.to_dict())

        return GraphServiceResponse(
            result={
                "twin": twin.to_dict(),
                "risk_evolution": risk_evo,
                "evolution_score": evolution,
            },
            execution_time_ms=timer.elapsed_ms,
            correlation_id=cid,
            confidence=risk_evo.get("prediction_confidence", 0.5),
            cache_hit=cached is not None,
            metadata={"incremental": existing is not None},
        )

    async def get_twin_summary(
        self,
        user_id: str,
        *,
        requester_id: str,
        window_days: int = 30,
    ) -> GraphServiceResponse[dict[str, Any]]:
        update = await self.update_twin(user_id, requester_id=requester_id)
        twin_data = update.result["twin"]
        twin = self._twins[user_id]
        story = GraphStoryGenerator.generate(
            twin, window_days=window_days, risk_evolution=update.result["risk_evolution"]
        )
        attention = AttentionVisualizer.compute(
            emotions=await self._fetch_emotions(user_id),
            symptoms=await self._fetch_symptoms(user_id),
            behaviours=await self._fetch_behaviours(user_id),
            sessions=await self._fetch_sessions(user_id),
            recommendations=await self._fetch_recommendations(user_id),
        )
        summaries = await self._longitudinal_summaries(user_id, twin)

        return GraphServiceResponse(
            result={
                "twin": twin_data,
                "story": story,
                "attention": attention,
                "risk_evolution": update.result["risk_evolution"],
                "evolution_score": update.result["evolution_score"],
                "longitudinal_summaries": summaries,
                "recommendation_timeline": await self._fetch_recommendations(user_id),
            },
            execution_time_ms=update.execution_time_ms,
            correlation_id=update.correlation_id,
            confidence=update.confidence,
        )

    async def capture_snapshot(
        self,
        user_id: str,
        *,
        requester_id: str,
        granularity: str = "daily",
    ) -> GraphServiceResponse[dict[str, Any]]:
        update = await self.update_twin(user_id, requester_id=requester_id)
        twin = self._twins[user_id]
        gran = TwinSnapshotGranularity(granularity)
        snap = self._snapshots.capture(twin, update.result["evolution_score"], gran)
        return GraphServiceResponse(
            result=snap.to_dict(),
            correlation_id=update.correlation_id,
            confidence=update.confidence,
        )

    async def dashboard_timelines(self, user_id: str, *, requester_id: str) -> GraphServiceResponse[dict[str, Any]]:
        if not GraphSecurityPolicy.can_read(requester_id, user_id):
            raise PermissionError("Unauthorized")
        with TimedOperation() as timer:
            return GraphServiceResponse(
                result={
                    "risk_timeline": await self._fetch_risk_timeline(user_id),
                    "emotion_timeline": await self._fetch_emotions(user_id),
                    "behaviour_timeline": await self._fetch_behaviours(user_id),
                    "symptom_timeline": await self._fetch_symptoms(user_id),
                    "recommendation_timeline": await self._fetch_recommendations(user_id),
                    "recovery_timeline": await self._recovery_timeline(user_id),
                },
                execution_time_ms=timer.elapsed_ms,
                correlation_id=str(uuid.uuid4()),
            )

    async def _fetch_user_data(self, user_id: str) -> dict[str, Any]:
        risks = await self._fetch_risk_timeline(user_id)
        risk_scores = [float(r.get("risk", 0)) for r in risks]
        return {
            "risk_scores": risk_scores,
            "emotions": await self._fetch_emotions(user_id),
            "symptoms": await self._fetch_symptoms(user_id),
            "behaviours": await self._fetch_behaviours(user_id),
            "sessions": await self._fetch_sessions(user_id),
            "recommendations": await self._fetch_recommendations(user_id),
            "user_props": {},
        }

    async def _fetch_risk_timeline(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = tq.risk_trend(user_id)
        records, _ = await self._repo._run(cypher, params)
        if not records:
            return []
        risks = records[0].get("risks", [])
        timestamps = records[0].get("timestamps", [])
        return [{"risk": r, "ts": t} for r, t in zip(risks, timestamps, strict=False)]

    async def _fetch_emotions(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = tq.emotion_trend(user_id)
        records, _ = await self._repo._run(cypher, params)
        return list(records)

    async def _fetch_symptoms(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = tq.symptom_trend(user_id)
        records, _ = await self._repo._run(cypher, params)
        return list(records)

    async def _fetch_behaviours(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = tq.behaviour_trend(user_id)
        records, _ = await self._repo._run(cypher, params)
        return list(records)

    async def _fetch_sessions(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = sq.session_timeline(user_id)
        records, _ = await self._repo._run(cypher, params)
        return list(records)

    async def _fetch_recommendations(self, user_id: str) -> list[dict[str, Any]]:
        cypher, params = rq.recommendation_history(user_id)
        records, _ = await self._repo._run(cypher, params)
        return list(records)

    async def _recovery_timeline(self, user_id: str) -> list[dict[str, Any]]:
        risks = await self._fetch_risk_timeline(user_id)
        timeline = []
        for i in range(1, len(risks)):
            delta = float(risks[i].get("risk", 0)) - float(risks[i - 1].get("risk", 0))
            timeline.append({
                "ts": risks[i].get("ts"),
                "risk_delta": round(delta, 4),
                "direction": "improving" if delta < 0 else "declining" if delta > 0 else "stable",
            })
        return timeline

    async def _longitudinal_summaries(self, user_id: str, twin: DigitalTwin) -> dict[str, Any]:
        windows = self._config.twin.get("summaries", {}).get("windows_days", [30, 90, 180, 365])
        summaries = {}
        for days in windows:
            since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            cypher, params = sq.sessions_in_window(user_id, since)
            records, _ = await self._repo._run(cypher, params)
            summaries[f"{days}_day"] = {
                "sessions": len(records),
                "engagement": twin.interaction.engagement_score,
                "dominant_emotion": twin.emotion.dominant_emotion,
                "risk_trend": "improving" if twin.recovery.risk_reduction > 0 else "stable",
                "mood_volatility": twin.emotion.variability,
            }
        return summaries
