"""PyTorch-compatible graph dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ml_pipeline.graph.export.exporters import to_pyg_dict
from ml_pipeline.graph.graph_embeddings.node2vec import graph_embedding
from ml_pipeline.graph.schema.types import GraphSnapshot


@dataclass
class GraphData:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    adj: torch.Tensor
    kg_vector: torch.Tensor
    num_nodes: int
    node_ids: list[str]
    metadata: dict[str, Any]


class GraphDataset:
  """Dataset of graph snapshots for GNN training."""

  def __init__(self, snapshots: list[GraphSnapshot], kg_dim: int = 64, feature_dim: int = 128) -> None:
      self.snapshots = snapshots
      self.kg_dim = kg_dim
      self.feature_dim = feature_dim

  def __len__(self) -> int:
      return len(self.snapshots)

  def __getitem__(self, idx: int) -> GraphData:
      return self.snapshot_to_data(self.snapshots[idx])

  def snapshot_to_data(self, snapshot: GraphSnapshot) -> GraphData:
      emb = graph_embedding(snapshot, dim=self.feature_dim)
      node_ids = [n.node_id for n in snapshot.nodes]
      n = len(node_ids)
      if n == 0:
          return GraphData(
              node_features=torch.zeros(1, self.feature_dim),
              edge_index=torch.zeros(2, 0, dtype=torch.long),
              adj=torch.eye(1),
              kg_vector=torch.zeros(self.kg_dim),
              num_nodes=1,
              node_ids=["empty"],
              metadata={},
          )
      feats = np.stack([emb.get(nid, np.zeros(self.feature_dim, dtype=np.float32)) for nid in node_ids])
      pyg = to_pyg_dict(snapshot, feats)
      edge_index = torch.as_tensor(pyg["edge_index"], dtype=torch.long)
      adj = torch.zeros(n, n)
      if edge_index.numel() > 0:
          adj[edge_index[0], edge_index[1]] = 1.0
          adj[edge_index[1], edge_index[0]] = 1.0
      kg = feats.mean(axis=0)[: self.kg_dim]
      if kg.shape[0] < self.kg_dim:
          kg = np.pad(kg, (0, self.kg_dim - kg.shape[0]))
      return GraphData(
          node_features=torch.as_tensor(feats, dtype=torch.float32),
          edge_index=edge_index,
          adj=adj,
          kg_vector=torch.as_tensor(kg, dtype=torch.float32),
          num_nodes=n,
          node_ids=node_ids,
          metadata=snapshot.metadata,
      )

  def collate_kg_batch(self, items: list[GraphData]) -> tuple[torch.Tensor, torch.Tensor]:
      """Return (kg_vectors B x kg_dim), (adj B x Nmax x Nmax) for batched GNN."""
      kg = torch.stack([it.kg_vector for it in items])
      max_n = max(it.num_nodes for it in items)
      adj_batch = torch.zeros(len(items), max_n, max_n)
      for i, it in enumerate(items):
          adj_batch[i, : it.num_nodes, : it.num_nodes] = it.adj
      return kg, adj_batch
