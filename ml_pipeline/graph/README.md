# Temporal Knowledge Graph (MP3 Part 4)

The graph is an **active reasoning component** in the MindGraph++ pipeline — not a visualization layer.

## Pipeline

```
Session → Graph Builder → Neo4j / Memory Store → Embeddings → GNN → Prediction
```

## Layout

| Path | Purpose |
|------|---------|
| `schema/` | Node/edge types (User, Session, Emotion, Symptom, …) |
| `builders/` | Neo4j, Temporal, Clinical, Research builders |
| `graph_store/` | In-memory + Neo4j stores |
| `gnn/` | GraphSAGE, GCN, GAT, GATv2, TGCN, Dynamic GraphSAGE |
| `graph_embeddings/` | Node2Vec, FastRP |
| `temporal/` | Multi-scale temporal encoding |
| `reasoning/` | Longitudinal trend rules |
| `queries/cypher_library.py` | Reusable Cypher modules |
| `dataset/` | PyG-compatible `GraphDataset` |
| `evaluation/` | Integrity validation |
| `export/` | JSON, CSV, GraphML, PyG |

## Usage

```python
from ml_pipeline.graph import GraphPipeline, build_gnn

pipe = GraphPipeline(builder_name="temporal")
data = pipe.build_tensors(session_data)  # kg_vector, adj
gnn = build_gnn("graphsage", 512, config)
out = gnn(fused_embedding, data.adj.unsqueeze(0))
```

## Commands

```bash
make graph-build      # Build graphs from feature store
make graph-validate   # Validate exported snapshots
make graph-benchmark  # Benchmark GNN architectures
```

## Neo4j Migrations

Apply `graphs/migrations/001_schema.cypher` then `002_relationships.cypher`.
