"""Graph attention weights for explainability visualization."""

from __future__ import annotations

from typing import Any


class AttentionVisualizer:
    """Generate node/edge/temporal attention maps for UI heatmaps."""

    @staticmethod
    def compute(
        *,
        emotions: list[dict[str, Any]],
        symptoms: list[dict[str, Any]],
        behaviours: list[dict[str, Any]],
        sessions: list[dict[str, Any]],
        recommendations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        node_attention = AttentionVisualizer._node_weights(emotions, symptoms, behaviours, sessions)
        edge_attention = AttentionVisualizer._edge_weights(emotions, symptoms)
        temporal_attention = AttentionVisualizer._temporal_weights(sessions)
        rec_attention = AttentionVisualizer._recommendation_weights(recommendations)

        return {
            "node_attention": node_attention,
            "edge_attention": edge_attention,
            "temporal_attention": temporal_attention,
            "recommendation_attention": rec_attention,
            "heatmap": AttentionVisualizer._heatmap(node_attention),
        }

    @staticmethod
    def _node_weights(
        emotions: list[dict],
        symptoms: list[dict],
        behaviours: list[dict],
        sessions: list[dict],
    ) -> list[dict[str, Any]]:
        weights: list[dict[str, Any]] = []
        for e in emotions[-5:]:
            weights.append({
                "label": "Emotion",
                "id": e.get("emotion", "unknown"),
                "weight": float(e.get("prob", 0)),
            })
        for s in symptoms[-5:]:
            weights.append({
                "label": "Symptom",
                "id": s.get("symptom", "unknown"),
                "weight": float(s.get("conf", s.get("severity", 0))),
            })
        for b in behaviours[-3:]:
            weights.append({
                "label": "Behaviour",
                "id": b.get("behaviour", b.get("behaviour_name", "unknown")),
                "weight": float(b.get("value", 0)),
            })
        for sess in sessions[-3:]:
            weights.append({
                "label": "Session",
                "id": sess.get("session_id", "unknown"),
                "weight": float(sess.get("quality_score", sess.get("quality", 0.5))),
            })
        return sorted(weights, key=lambda x: x["weight"], reverse=True)

    @staticmethod
    def _edge_weights(emotions: list[dict], symptoms: list[dict]) -> list[dict[str, Any]]:
        edges = []
        if emotions and symptoms:
            edges.append({
                "source": "Emotion",
                "target": "Symptom",
                "type": "CORRELATES_WITH",
                "weight": 0.7,
            })
        return edges

    @staticmethod
    def _temporal_weights(sessions: list[dict]) -> list[dict[str, Any]]:
        return [
            {"session_id": s.get("session_id", f"s{i}"), "weight": (i + 1) / max(len(sessions), 1)}
            for i, s in enumerate(sessions[-10:])
        ]

    @staticmethod
    def _recommendation_weights(recommendations: list[dict]) -> list[dict[str, Any]]:
        return [
            {
                "recommendation_id": r.get("recommendation_id", r.get("id", "")),
                "type": r.get("recommendation_type", r.get("type", "")),
                "weight": float(r.get("confidence", 0.5)),
            }
            for r in recommendations[:5]
        ]

    @staticmethod
    def _heatmap(node_attention: list[dict]) -> dict[str, float]:
        return {f"{n['label']}:{n['id']}": n["weight"] for n in node_attention[:10]}
