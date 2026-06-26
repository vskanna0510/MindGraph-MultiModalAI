"""Neo4j-oriented graph builder."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ml_pipeline.graph.builders.base import BaseGraphBuilder
from ml_pipeline.graph.schema.types import GraphEdge, GraphNode, GraphSnapshot


class Neo4jGraphBuilder(BaseGraphBuilder):
    """Build user → session → feature → prediction graph."""

    def build(self, data: dict[str, Any]) -> GraphSnapshot:
        user_id = str(data["participant_id"])
        session_id = str(data.get("session_id", user_id))
        dataset = str(data.get("dataset", "daic"))

        user = GraphNode.create("User", "user_id", user_id, created_at=datetime.now(UTC).isoformat(), language=data.get("language", "en"))
        self.store.upsert_node(user)

        session = GraphNode.create(
            "Session",
            "session_id",
            session_id,
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            quality_score=float(data.get("quality_score", 0.0)),
            dataset=dataset,
        )
        self.store.upsert_node(session)
        self.store.add_edge(GraphEdge(user.node_id, session.node_id, "HAS_SESSION"))

        ds = GraphNode.create("Dataset", "dataset_id", dataset, name=dataset)
        self.store.upsert_node(ds)
        self.store.add_edge(GraphEdge(session.node_id, ds.node_id, "PART_OF_DATASET"))

        for mod in ("audio", "visual", "text", "image"):
            if not data.get("modalities", {}).get(mod, data.get(f"has_{mod}", False)):
                continue
            label = f"{mod.capitalize()}Feature" if mod != "audio" else "AudioFeature"
            if mod == "visual":
                label = "VisualFeature"
            elif mod == "text":
                label = "TextFeature"
            elif mod == "image":
                label = "ImageFeature"
            feat_id = f"{session_id}_{mod}"
            feat = GraphNode.create(
                label,
                "feature_id",
                feat_id,
                feature_version=data.get("feature_version", "v1"),
                quality_score=float(data.get("quality", {}).get(mod, 0.8)),
                dimension=int(data.get("embedding_dims", {}).get(mod, 128)),
            )
            self.store.upsert_node(feat)
            self.store.add_edge(GraphEdge(session.node_id, feat.node_id, "HAS_FEATURE"))

        if "label" in data:
            pred = GraphNode.create(
                "Prediction",
                "prediction_id",
                f"pred_{session_id}",
                prediction_label=str(data["label"]),
                confidence=float(data.get("prediction_confidence", 0.5)),
                model_version=data.get("model_version", "MODEL_010"),
            )
            self.store.upsert_node(pred)
            self.store.add_edge(GraphEdge(session.node_id, pred.node_id, "PREDICTED"))

        return self.store.snapshot(metadata={"builder": "neo4j", "user_id": user_id})
