"""MindGraph++ temporal knowledge graph pipeline."""

from ml_pipeline.graph.pipeline import GraphPipeline
from ml_pipeline.graph.builders import build_graph_builder
from ml_pipeline.graph.dataset.graph_dataset import GraphData, GraphDataset
from ml_pipeline.graph.config import graph_config, graph_paths

__all__ = [
    "GraphPipeline",
    "build_graph_builder",
    "GraphData",
    "GraphDataset",
    "graph_config",
    "graph_paths",
]

def build_gnn(*args, **kwargs):
    from ml_pipeline.graph.gnn.registry import build_gnn as _build

    return _build(*args, **kwargs)


def list_gnn_architectures():
    from ml_pipeline.graph.gnn.registry import list_gnn_architectures as _list

    return _list()
