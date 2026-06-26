"""Graph pipeline unit tests (MP3 Part 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml_pipeline.graph.analytics.metrics import compute_analytics
from ml_pipeline.graph.builders import build_graph_builder
from ml_pipeline.graph.builders.temporal_builder import TemporalGraphBuilder
from ml_pipeline.graph.cache.graph_cache import GraphCache
from ml_pipeline.graph.dataset.graph_dataset import GraphDataset
from ml_pipeline.graph.evaluation.validator import GraphValidator
from ml_pipeline.graph.export.exporters import export_csv, export_graphml, export_json, to_pyg_dict
from ml_pipeline.graph.gnn import build_gnn, list_gnn_architectures
from ml_pipeline.graph.graph_embeddings.node2vec import graph_embedding, random_walk
from ml_pipeline.graph.pipeline import GraphPipeline
from ml_pipeline.graph.queries.cypher_library import risk_progression, user_timeline
from ml_pipeline.graph.reasoning.trends import assess_risk_trend, reason_over_history
from ml_pipeline.graph.schema.types import GraphEdge, GraphNode, GraphSnapshot
from ml_pipeline.models.graph.gnn import KnowledgeGraphInjection
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.models.types import MultimodalBatch


@pytest.fixture
def session_data():
    return {
        "participant_id": "P001",
        "session_id": "S001",
        "dataset": "daic",
        "language": "en",
        "label": "depression",
        "quality_score": 0.85,
        "modalities": {"audio": True, "text": True, "visual": False, "image": False},
    }


def test_neo4j_builder(session_data):
    builder = build_graph_builder("neo4j")
    snap = builder.build(session_data)
    assert snap.node_count >= 3
    assert any(n.label == "User" for n in snap.nodes)


def test_temporal_builder_links_sessions(session_data):
    builder = TemporalGraphBuilder()
    builder.build(session_data)
    data2 = {**session_data, "session_id": "S002"}
    snap = builder.build(data2)
    temporal = [e for e in snap.edges if e.rel_type == "TEMPORALLY_PRECEDES"]
    assert len(temporal) >= 1


def test_clinical_builder_emotions(session_data):
    builder = build_graph_builder("clinical")
    data = {**session_data, "emotions": {"Sadness": 0.8, "Stress": 0.6}}
    snap = builder.build(data)
    assert any(n.label == "Emotion" for n in snap.nodes)


def test_graph_validator(session_data):
    snap = build_graph_builder("neo4j").build(session_data)
    ok, errs = GraphValidator().validate(snap)
    assert ok or "disconnected_session_chain" in errs


def test_node2vec_embedding(session_data):
    snap = build_graph_builder("temporal").build(session_data)
    emb = graph_embedding(snap, dim=32)
    assert len(emb) == snap.node_count


def test_graph_dataset_tensors(session_data):
    snap = build_graph_builder("temporal").build(session_data)
    ds = GraphDataset([snap], kg_dim=64)
    data = ds[0]
    assert data.kg_vector.shape == (64,)
    assert data.adj.dim() == 2


def test_export_formats(session_data, tmp_path: Path):
    snap = build_graph_builder("temporal").build(session_data)
    export_json(snap, tmp_path / "g.json")
    export_graphml(snap, tmp_path / "g.graphml")
    export_csv(snap, tmp_path / "csv")
    pyg = to_pyg_dict(snap)
    assert "edge_index" in pyg


@pytest.mark.parametrize("arch", ["graphsage", "gcn", "gat", "gatv2", "tgcn", "dynamic_graphsage"])
def test_gnn_forward(arch):
    gnn = build_gnn(arch, 128, {"hidden_dim": 64, "num_layers": 1, "heads": 2})
    # Node-level graph (1 graph, 4 nodes)
    x = torch.randn(1, 4, 128)
    adj = torch.ones(1, 4, 4)
    try:
        out = gnn(x, adj)
    except TypeError:
        out = gnn(x, adj, torch.zeros(1))
    assert out.shape[0] == 1
    # Fused embedding path (no adj)
    x2 = torch.randn(2, 128)
    out2 = gnn(x2, None)
    assert out2.shape == (2, 128)


def test_kg_injection():
    inj = KnowledgeGraphInjection(128, kg_dim=64)
    fused = torch.randn(2, 128)
    kg = torch.randn(2, 64)
    out = inj(fused, kg)
    assert out.shape == (2, 128)


def test_kg_injection_none():
    inj = KnowledgeGraphInjection(128)
    fused = torch.randn(2, 128)
    assert torch.equal(inj(fused, None), fused)


def test_reasoning_trends():
    assert assess_risk_trend([0.2, 0.5, 0.8]) == "risk_increasing"
    assert assess_risk_trend([0.8, 0.5, 0.2]) == "risk_decreasing"
    out = reason_over_history([0.3, 0.5, 0.7])
    assert out["deterioration"] is True


def test_cypher_library():
    q, p = user_timeline("P001")
    assert "User" in q and p["user_id"] == "P001"
    q2, _ = risk_progression("P001")
    assert "Prediction" in q2


def test_graph_cache(tmp_path: Path):
    cache = GraphCache(tmp_path)
    cache.set("emb", "P001", {"vec": torch.randn(64).numpy()})
    got = cache.get("emb", "P001")
    assert got is not None
    assert cache.invalidate("emb") >= 1


def test_graph_pipeline(session_data):
    pipe = GraphPipeline()
    data = pipe.build_tensors(session_data)
    assert data.kg_vector.numel() == 64


def test_analytics(session_data):
    snap = build_graph_builder("temporal").build(session_data)
    stats = compute_analytics(snap)
    assert "node_count" in stats


def test_mindgraph_with_graph_tensor():
    batch = MultimodalBatch(
        audio=torch.randn(2, 8, 768),
        text=torch.randn(2, 8, 768),
        graph=torch.randn(2, 64),
        audio_mask=torch.ones(2, 8),
        text_mask=torch.ones(2, 8),
        modality_presence={"audio": torch.ones(2), "text": torch.ones(2), "visual": torch.zeros(2), "image": torch.zeros(2)},
        metadata={},
        labels=torch.tensor([0, 1]),
    )
    model = MindGraphMultimodal()
    out = model.forward(batch)
    assert out["fused"].shape[0] == 2


def test_gnn_registry():
    assert "graphsage" in list_gnn_architectures()
