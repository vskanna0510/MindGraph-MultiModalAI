"""End-to-end graph construction pipeline."""

from __future__ import annotations

from typing import Any

from ml_pipeline.graph.builders import build_graph_builder
from ml_pipeline.graph.cache.graph_cache import GraphCache
from ml_pipeline.graph.config import graph_config, graph_paths
from ml_pipeline.graph.dataset.graph_dataset import GraphData, GraphDataset
from ml_pipeline.graph.evaluation.validator import GraphValidator
from ml_pipeline.graph.export.exporters import export_json
from ml_pipeline.graph.schema.types import GraphSnapshot


class GraphPipeline:
    """Session → graph build → embedding → tensor bridge."""

    def __init__(self, builder_name: str = "temporal") -> None:
        cfg = graph_config()
        self.cfg = cfg
        self.builder_name = builder_name
        self.builder = build_graph_builder(builder_name)
        self.cache = GraphCache()
        self.kg_dim = int(cfg.get("pipeline", {}).get("kg_dim", 64))
        self.paths = graph_paths()

    def build_session_graph(self, session_data: dict[str, Any]) -> GraphSnapshot:
        snap = self.builder.build(session_data)
        ok, errs = GraphValidator().validate(snap)
        if not ok:
            snap.metadata["validation_errors"] = errs
        return snap

    def to_tensors(self, snapshot: GraphSnapshot) -> GraphData:
        ds = GraphDataset([snapshot], kg_dim=self.kg_dim)
        return ds[0]

    def build_tensors(self, session_data: dict[str, Any]) -> GraphData:
        snap = self.build_session_graph(session_data)
        data = self.to_tensors(snap)
        cache_id = f"{session_data.get('participant_id')}:{session_data.get('session_id')}"
        self.cache.set("tensors", cache_id, {"kg_vector": data.kg_vector.numpy()})
        return data

    def export_snapshot(self, snapshot: GraphSnapshot, name: str) -> None:
        export_json(snapshot, self.paths["exports"] / f"{name}.json")
