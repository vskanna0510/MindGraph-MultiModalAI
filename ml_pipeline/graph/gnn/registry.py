"""GNN factory registry."""

from __future__ import annotations

from typing import Any, Type

from ml_pipeline.graph.gnn.base import BaseGraphModule

from ml_pipeline.graph.gnn.dynamic_graphsage import DynamicGraphSAGEModule
from ml_pipeline.graph.gnn.gat import GATModule
from ml_pipeline.graph.gnn.gatv2 import GATv2Module
from ml_pipeline.graph.gnn.gcn import GCNModule
from ml_pipeline.graph.gnn.graphsage import GraphSAGEModule
from ml_pipeline.graph.gnn.tgcn import TGCNModule

GNN_REGISTRY: dict[str, Type[BaseGraphModule]] = {
    "graphsage": GraphSAGEModule,
    "gcn": GCNModule,
    "gat": GATModule,
    "gatv2": GATv2Module,
    "tgcn": TGCNModule,
    "dynamic_graphsage": DynamicGraphSAGEModule,
}


def list_gnn_architectures() -> list[str]:
    return sorted(GNN_REGISTRY.keys())


def build_gnn(architecture: str, in_dim: int, config: dict[str, Any] | None = None) -> BaseGraphModule:
    cfg = config or {}
    key = architecture.lower().replace("-", "_")
    if key not in GNN_REGISTRY:
        raise ValueError(f"Unknown GNN '{architecture}'. Available: {list_gnn_architectures()}")
    kwargs: dict[str, Any] = {
        "in_dim": in_dim,
        "hidden_dim": int(cfg.get("hidden_dim", 256)),
        "num_layers": int(cfg.get("num_layers", 2)),
        "dropout": float(cfg.get("dropout", 0.1)),
    }
    if key in ("gat", "gatv2"):
        kwargs["heads"] = int(cfg.get("heads", 4))
    return GNN_REGISTRY[key](**kwargs)
