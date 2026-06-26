"""Research graph builder — study and model version nodes."""

from __future__ import annotations

from typing import Any

from ml_pipeline.graph.builders.clinical_builder import ClinicalGraphBuilder
from ml_pipeline.graph.schema.types import GraphEdge, GraphNode, GraphSnapshot


class ResearchGraphBuilder(ClinicalGraphBuilder):
    def build(self, data: dict[str, Any]) -> GraphSnapshot:
        snap = super().build(data)
        study_id = str(data.get("study_id", "mindgraph_pp"))
        model_ver = str(data.get("model_version", "MODEL_010"))

        study = GraphNode.create("ResearchStudy", "study_id", study_id, name=study_id)
        self.store.upsert_node(study)

        mv = GraphNode.create("ModelVersion", "model_id", model_ver, version=model_ver)
        self.store.upsert_node(mv)

        user_nid = f"User:{data['participant_id']}"
        self.store.add_edge(GraphEdge(user_nid, study.node_id, "BELONGS_TO"))
        self.store.add_edge(GraphEdge(user_nid, mv.node_id, "USES_MODEL"))

        return self.store.snapshot(metadata={"builder": "research", "study_id": study_id})
