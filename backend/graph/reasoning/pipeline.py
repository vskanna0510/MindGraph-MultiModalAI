"""7-layer graph reasoning pipeline."""

from __future__ import annotations

import uuid
from typing import Any

from neo4j import AsyncSession

from graph.cache import GraphCache
from graph.config.loader import get_graph_config
from graph.contracts import GraphServiceResponse, TimedOperation
from graph.reasoning.alerts import AlertEngine
from graph.reasoning.anomaly import AnomalyDetector
from graph.reasoning.patterns import PatternDetector
from graph.reasoning.retrieval import GraphRetrievalEngine, TemporalWindow
from graph.reasoning.risk_trend import RiskTrendEngine
from graph.reasoning.similarity import GraphSimilarityEngine
from graph.reasoning.temporal_reasoning import TemporalReasoningEngine
from graph.repositories.base import BaseGraphRepository
from graph.queries import prediction_queries as pq
from graph.queries import trend_queries as tq
from graph.validators import GraphValidator


class GraphReasoningPipeline:
    """Full reasoning pipeline: retrieval → temporal → analytics → recommendations → explainability."""

    def __init__(self, session: AsyncSession, *, cache: GraphCache | None = None) -> None:
        self._session = session
        self._repo = BaseGraphRepository(session)
        self._cache = cache or GraphCache()
        self._retrieval = GraphRetrievalEngine(session, cache=self._cache)
        self._temporal = TemporalReasoningEngine()
        self._alerts = AlertEngine(session)
        from graph.services.explainability_service import ExplainabilityService
        from graph.services.recommendation_service import RecommendationService

        self._recommendations = RecommendationService(session, cache=self._cache)
        self._explain = ExplainabilityService(session)
        self._validator = GraphValidator()
        self._config = get_graph_config()

    async def reason(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        window: TemporalWindow = TemporalWindow.LAST_30_DAYS,
        correlation_id: str | None = None,
    ) -> GraphServiceResponse[dict[str, Any]]:
        cid = correlation_id or str(uuid.uuid4())
        warnings: list[str] = []

        with TimedOperation() as timer:
            # Pre-reasoning validation
            validation = await self._validate_graph(user_id)
            if not validation["valid"]:
                warnings.extend(validation["errors"])

            # Layer 1: Entity retrieval
            user_graph = await self._retrieval.user_graph(user_id, hops=2, correlation_id=cid)
            sessions = await self._retrieval.temporal_window_sessions(user_id, window, correlation_id=cid)

            # Layer 2: Relationship expansion
            subgraph = None
            if session_id:
                subgraph = await self._retrieval.session_graph(session_id, hops=2, correlation_id=cid)

            # Layer 3: Temporal context — risk scores
            cypher, params = tq.risk_trend(user_id)
            risk_records, _ = await self._repo._run(cypher, params)
            risk_scores = list(risk_records[0].get("risks", [])) if risk_records else []

            temporal_analysis = self._temporal.analyze_risk([float(s) for s in risk_scores])

            # Layer 4: Graph analytics
            risk_metrics = RiskTrendEngine.to_dict(RiskTrendEngine.compute([float(s) for s in risk_scores]))

            cypher_e, params_e = tq.emotion_trend(user_id)
            emotion_records, _ = await self._repo._run(cypher_e, params_e)
            emotions = list(emotion_records)
            neg_probs = [
                float(e.get("prob", 0))
                for e in emotions
                if str(e.get("emotion", "")).lower() in {"sadness", "hopelessness", "stress", "fear"}
            ]

            cypher_s, params_s = tq.symptom_trend(user_id)
            symptom_records, _ = await self._repo._run(cypher_s, params_s)

            patterns = {
                "persistent_symptoms": PatternDetector.persistent_symptoms(symptom_records),
                "escalation": PatternDetector.escalation_pattern([float(s) for s in risk_scores]),
                "recovery": PatternDetector.recovery_pattern([float(s) for s in risk_scores]),
            }

            anomalies = AnomalyDetector.scan([float(s) for s in risk_scores], neg_probs or None)

            # Layer 5: Historical comparison
            historical = {
                "risk_metrics": risk_metrics,
                "temporal_state": temporal_analysis,
                "session_count": len(sessions.result),
                "emotion_stability": self._temporal.analyze_emotion_stability(
                    [float(e.get("prob", 0)) for e in emotions]
                ),
            }

            # Layer 6: Recommendations
            recs = await self._recommendations.for_user(user_id)
            if session_id and risk_scores:
                suggested = await self._recommendations.suggest_from_risk(
                    session_id, user_id, owner=user_id, risk_score=float(risk_scores[-1])
                )
            else:
                suggested = []

            # Layer 7: Explainability
            explanation = await self._explain.explain_user_state(user_id)
            if risk_scores and len(risk_scores) >= 2:
                explanation["why_changed"] = (
                    f"Risk changed by {float(risk_scores[-1]) - float(risk_scores[-2]):.3f} "
                    f"from prior session. State: {temporal_analysis['state']}."
                )

            # Alerts
            alerts = await self._alerts.evaluate(
                user_id,
                [float(s) for s in risk_scores],
                negative_emotion_probs=neg_probs or None,
            )

            # Prediction refinement
            refined_confidence = min(1.0, temporal_analysis.get("confidence", 0.5) + 0.1)

            result = {
                "user_id": user_id,
                "session_id": session_id,
                "layers": {
                    "entity_retrieval": {"nodes": len(user_graph.result), "cache_hit": user_graph.cache_hit},
                    "relationship_expansion": {"subgraph_paths": len(subgraph.result) if subgraph else 0},
                    "temporal_context": temporal_analysis,
                    "graph_analytics": {"risk_metrics": risk_metrics, "patterns": patterns, "anomalies": anomalies},
                    "historical_comparison": historical,
                    "recommendations": {"existing": recs, "suggested": suggested},
                    "explainability": explanation,
                },
                "alerts": alerts,
                "refined_confidence": refined_confidence,
            }

        return GraphServiceResponse(
            result=result,
            execution_time_ms=timer.elapsed_ms,
            correlation_id=cid,
            confidence=refined_confidence,
            warnings=warnings,
            version=self._config.reasoning.get("reasoning", {}).get("version", "2.0.0"),
            metadata={"window": window.value, "pipeline_layers": 7},
        )

    async def _validate_graph(self, user_id: str) -> dict[str, Any]:
        cypher, params = pq.prediction_history(user_id, limit=100)
        records, _ = await self._repo._run(cypher, params)
        timestamps = [str(r.get("ts", "")) for r in records if r.get("ts")]
        order_result = self._validator.validate_temporal_order(timestamps) if len(timestamps) >= 2 else None
        errors = list(order_result.errors) if order_result and not order_result.valid else []
        return {"valid": len(errors) == 0, "errors": errors, "predictions": len(records)}
