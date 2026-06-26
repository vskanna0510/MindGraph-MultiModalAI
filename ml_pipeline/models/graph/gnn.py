"""Graph neural network and knowledge graph injection."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml_pipeline.graph.gnn.graphsage import GraphSAGEModule
from ml_pipeline.graph.gnn.registry import build_gnn, list_gnn_architectures
from ml_pipeline.graph.gnn.base import BaseGraphModule

__all__ = ["KnowledgeGraphInjection", "GraphSAGEModule", "build_gnn", "list_gnn_architectures", "BaseGraphModule"]


class KnowledgeGraphInjection(nn.Module):
    """Inject external graph context into fused representation."""

    def __init__(self, d_model: int, kg_dim: int = 64) -> None:
        super().__init__()
        self.kg_proj = nn.Linear(kg_dim, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, fused: torch.Tensor, kg_vector: torch.Tensor | None) -> torch.Tensor:
        if kg_vector is None:
            return fused
        kg = self.kg_proj(kg_vector)
        gate = self.gate(torch.cat([fused, kg], dim=-1))
        return fused * (1 - gate) + kg * gate
