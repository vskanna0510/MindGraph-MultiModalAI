"""Fusion architecture unit tests (MP3 Part 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml_pipeline.models.fusion.attention_pooling import FusionAttentionPooling
from ml_pipeline.models.fusion.benchmark import benchmark_all_strategies
from ml_pipeline.models.fusion.cross_attention import CrossAttentionFusion
from ml_pipeline.models.fusion.early_fusion import EarlyFusion
from ml_pipeline.models.fusion.fusion_registry import MultimodalFusionStack, build_fusion, list_fusion_strategies
from ml_pipeline.models.fusion.gated_fusion import DynamicFusion, GatedFusion
from ml_pipeline.models.fusion.late_fusion import LateFusion
from ml_pipeline.models.fusion.output import FusionOutput
from ml_pipeline.models.fusion.projection import FusionProjection, MultimodalProjection
from ml_pipeline.models.fusion.quality_aware import QualityAwareFusion
from ml_pipeline.models.fusion.residual_fusion import ResidualFusion
from ml_pipeline.models.fusion.temporal_fusion import TemporalFusion
from ml_pipeline.models.fusion.temporal_memory import TemporalMemory
from ml_pipeline.models.fusion.transformer_fusion import TransformerFusion
from ml_pipeline.models.fusion.visualization import export_fusion_report
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.models.types import MultimodalBatch


@pytest.fixture
def embeddings():
    b, d = 2, 128
    return {
        "audio": torch.randn(b, d),
        "text": torch.randn(b, d),
        "visual": torch.randn(b, d),
        "image": torch.randn(b, d),
    }


@pytest.fixture
def presence():
  b = 2
  return {
      "audio": torch.ones(b),
      "text": torch.ones(b),
      "visual": torch.zeros(b),
      "image": torch.zeros(b),
  }


def test_fusion_projection():
    proj = FusionProjection(256, 128)
    x = torch.randn(2, 256)
    out = proj(x)
    assert out.shape == (2, 128)


def test_multimodal_projection(embeddings):
    mp = MultimodalProjection(["audio", "text"], 128, 128)
    out = mp({"audio": embeddings["audio"], "text": embeddings["text"]})
    assert set(out.keys()) == {"audio", "text"}


@pytest.mark.parametrize(
    "strategy",
    ["early", "late", "cross_attention", "co_attention", "transformer", "gated", "dynamic", "residual", "quality_aware"],
)
def test_fusion_strategies(strategy, embeddings, presence):
    fusion = build_fusion(strategy, 128, {"num_heads": 4, "num_layers": 1})
    out = fusion.fuse(embeddings, presence)
    assert isinstance(out, FusionOutput)
    assert out.fusion_embedding.shape == (2, 128)
    ok, errs = fusion.validate()
    assert ok, errs


def test_missing_modalities(embeddings, presence):
    fusion = TransformerFusion(128, {"num_heads": 4, "num_layers": 1})
    out = fusion.fuse(embeddings, presence)
    assert out.fusion_embedding.shape == (2, 128)


def test_quality_aware_fusion(embeddings, presence):
    fusion = QualityAwareFusion(128)
    q = {"audio": torch.tensor([0.9, 0.8]), "text": torch.tensor([0.5, 0.6])}
    out = fusion.fuse(embeddings, presence, q)
    assert out.quality_weights


def test_temporal_memory():
    mem = TemporalMemory(64, max_sessions=4)
    cur = torch.randn(2, 64)
    hist = torch.randn(2, 3, 64)
    seq = mem.build_sequence(cur, hist)
    assert seq.shape == (2, 4, 64)


def test_temporal_fusion():
    tf = TemporalFusion(128, {"num_layers": 1, "max_sessions": 4})
    emb = torch.randn(2, 128)
    out = tf(emb)
    assert out.temporal_embedding is not None
    assert out.temporal_embedding.shape == (2, 128)


def test_fusion_stack():
    stack = MultimodalFusionStack(
        {"fusion": {"strategy": "transformer", "d_model": 128, "num_layers": 1}, "temporal": {"enabled": True}}
    )
    embs = {"audio": torch.randn(2, 128), "text": torch.randn(2, 128)}
    pres = {"audio": torch.ones(2), "text": torch.ones(2), "visual": torch.zeros(2), "image": torch.zeros(2)}
    out = stack(embs, pres)
    fused = stack.fused_tensor(out)
    assert fused.shape == (2, 128)


def test_attention_pooling():
    pool = FusionAttentionPooling(64)
    x = torch.randn(2, 5, 64)
    pooled, attn = pool(x)
    assert pooled.shape == (2, 64)
    assert attn is not None


def test_fusion_registry_lists():
    strategies = list_fusion_strategies()
    assert "transformer" in strategies
    assert "cross_modal_transformer" in strategies


def test_fusion_benchmark_smoke():
    results = benchmark_all_strategies(d_model=64, batch_size=2, runs=2, strategies=["early", "late", "transformer"])
    assert "early" in results
    assert "mean_ms" in results["early"] or "error" not in results["early"]


def test_fusion_visualization(tmp_path: Path, embeddings, presence):
    fusion = LateFusion(128)
    out = fusion.fuse(embeddings, presence)
    paths = export_fusion_report(out, tmp_path)
    assert any(p.name == "fusion_metadata.json" for p in paths)


def test_fusion_save_load(tmp_path: Path, embeddings, presence):
    fusion = EarlyFusion(128)
    path = fusion.save(tmp_path / "early.pt")
    assert path.exists()
    loaded = build_fusion("early", 128)
    ckpt = torch.load(path, weights_only=False)
    loaded.load_state_dict(ckpt["state_dict"])
    out = loaded.fuse(embeddings, presence)
    assert out.fusion_embedding.shape == (2, 128)


def test_mindgraph_with_fusion_stack():
    batch = MultimodalBatch(
        audio=torch.randn(2, 8, 768),
        text=torch.randn(2, 8, 768),
        audio_mask=torch.ones(2, 8),
        text_mask=torch.ones(2, 8),
        modality_presence={"audio": torch.ones(2), "visual": torch.zeros(2), "text": torch.ones(2), "image": torch.zeros(2)},
        labels=torch.tensor([0, 1]),
    )
    model = MindGraphMultimodal()
    out = model.forward(batch)
    assert "fusion_output" in out
    assert out["fused"].shape[0] == 2


def test_torchscript_projection():
    proj = FusionProjection(64, 64)
    proj.eval()
    traced = torch.jit.trace(proj, torch.randn(1, 64))
    assert traced(torch.randn(1, 64)).shape == (1, 64)
