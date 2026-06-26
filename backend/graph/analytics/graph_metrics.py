"""Graph metrics computation from retrieved subgraph data."""

from __future__ import annotations

from collections import Counter
from typing import Any


class GraphMetricsEngine:
    """Compute density, degree, centrality from node/edge lists."""

    @staticmethod
    def from_degree_map(degree_map: dict[str, int], node_count: int, edge_count: int) -> dict[str, Any]:
        if node_count == 0:
            return {"node_count": 0, "edge_count": 0, "density": 0.0, "avg_degree": 0.0}

        max_edges = node_count * (node_count - 1) or 1
        density = (2 * edge_count) / max_edges
        degrees = list(degree_map.values())
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

        ranked = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [{"id": k, "degree": v} for k, v in ranked[:10]]

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": round(density, 4),
            "avg_degree": round(avg_degree, 4),
            "max_degree": max(degrees) if degrees else 0,
            "top_central_nodes": top_nodes,
        }

    @staticmethod
    def node_importance(
        records: list[dict[str, Any]],
        *,
        label_key: str = "label",
        degree_key: str = "degree",
    ) -> list[dict[str, Any]]:
        ranked = sorted(records, key=lambda r: int(r.get(degree_key, 0)), reverse=True)
        return [
            {
                "label": r.get(label_key),
                "uuid": r.get("uuid"),
                "degree": r.get(degree_key),
                "importance_score": round(int(r.get(degree_key, 0)) / max(len(ranked), 1), 4),
            }
            for r in ranked
        ]

    @staticmethod
    def emotion_cooccurrence(emotion_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        emotions = [str(r.get("emotion", "")) for r in emotion_records if r.get("emotion")]
        counts = Counter(emotions)
        return [{"emotion": e, "frequency": c} for e, c in counts.most_common()]
