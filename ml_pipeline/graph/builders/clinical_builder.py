"""Clinical graph builder — symptoms, emotions (no diagnosis)."""

from __future__ import annotations

from typing import Any

from ml_pipeline.graph.builders.temporal_builder import TemporalGraphBuilder
from ml_pipeline.graph.schema.nodes import EMOTIONS, SYMPTOMS
from ml_pipeline.graph.schema.types import GraphEdge, GraphNode, GraphSnapshot


class ClinicalGraphBuilder(TemporalGraphBuilder):
    def build(self, data: dict[str, Any]) -> GraphSnapshot:
        snap = super().build(data)
        session_id = str(data.get("session_id", data["participant_id"]))
        session_nid = f"Session:{session_id}"

        for emotion, prob in (data.get("emotions") or {}).items():
            if emotion not in EMOTIONS:
                continue
            node = GraphNode.create("Emotion", "emotion", f"{session_id}_{emotion}", probability=float(prob), source_modality="multimodal")
            self.store.upsert_node(node)
            self.store.add_edge(GraphEdge(session_nid, node.node_id, "EXPRESSES", {"confidence": float(prob)}))

        for symptom, conf in (data.get("symptoms") or {}).items():
            if symptom not in SYMPTOMS:
                continue
            node = GraphNode.create("Symptom", "symptom", f"{session_id}_{symptom}", confidence=float(conf), source="inference")
            self.store.upsert_node(node)
            self.store.add_edge(GraphEdge(session_nid, node.node_id, "INDICATES", {"confidence": float(conf)}))

        return self.store.snapshot(metadata={"builder": "clinical"})
