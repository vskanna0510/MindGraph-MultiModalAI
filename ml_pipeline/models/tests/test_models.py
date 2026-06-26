"""Tests for multimodal model architecture (MP3 Part 1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml_pipeline.models.audio.encoder import AudioEncoder
from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.text.encoder import TextEncoder
from ml_pipeline.models.config import model_config
from ml_pipeline.models.device import resolve_device, resolve_dtype
from ml_pipeline.models.fusion.cross_modal import CrossModalTransformer
from ml_pipeline.models.losses.multimodal import MultimodalLoss
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.models.registry import ModelRegistry
from ml_pipeline.models.text.encoder import TextEncoder
from ml_pipeline.models.types import MultimodalBatch


@pytest.fixture
def batch() -> MultimodalBatch:
    b, t, d = 2, 8, 768
    return MultimodalBatch(
        audio=torch.randn(b, t, d),
        text=torch.randn(b, t, d),
        audio_mask=torch.ones(b, t),
        text_mask=torch.ones(b, t),
        modality_presence={
            "audio": torch.ones(b),
            "visual": torch.zeros(b),
            "text": torch.ones(b),
            "image": torch.zeros(b),
        },
        labels=torch.tensor([0, 1]),
    )


def test_model_config():
    cfg = model_config()
    assert cfg.get("model", {}).get("latent_dim")


def test_audio_encoder_forward():
    enc = AudioEncoder({"input_dim": 512, "latent_dim": 256, "hidden_dim": 256, "num_layers": 2})
    x = torch.randn(2, 10, 512)
    out = enc(x, torch.ones(2, 10))
    assert isinstance(out, EncoderOutput)
    assert out.embedding.shape == (2, 256)


def test_text_encoder_forward():
    enc = TextEncoder({"input_dim": 512, "latent_dim": 256, "hidden_dim": 256, "num_layers": 2})
    out = enc(torch.randn(2, 5, 512))
    assert isinstance(out, EncoderOutput)
    assert out.embedding.shape == (2, 256)


def test_cross_modal_fusion():
    fusion = CrossModalTransformer(256, 4, 1)
    embs = {"audio": torch.randn(2, 256), "text": torch.randn(2, 256)}
    pres = {"audio": torch.ones(2), "text": torch.ones(2)}
    out = fusion(embs, pres)
    assert out.shape == (2, 256)


def test_mindgraph_forward(batch: MultimodalBatch):
    model = MindGraphMultimodal()
    out = model.forward(batch)
    assert out["logits"].shape == (2, 2)
    assert out["fused"].shape[0] == 2


def test_mindgraph_backward(batch: MultimodalBatch):
    model = MindGraphMultimodal()
    out = model.forward(batch)
    loss_fn = MultimodalLoss()
    losses = loss_fn(out["logits"], batch.labels, out["risk_logits"])
    losses["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_predict(batch: MultimodalBatch):
    model = MindGraphMultimodal()
    out = model.predict(batch)
    assert "probs" in out
    assert "preds" in out


def test_extract_features(batch: MultimodalBatch):
    model = MindGraphMultimodal()
    feats = model.extract_features(batch)
    assert "fused" in feats


def test_save_load(batch: MultimodalBatch, tmp_path: Path):
    model = MindGraphMultimodal()
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = MindGraphMultimodal.load(path)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        out1 = model.forward(batch)
        out2 = loaded.forward(batch)
    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-5)


def test_validate():
    model = MindGraphMultimodal()
    ok, errs = model.validate()
    assert ok
    assert not errs


def test_summary_and_benchmark(batch: MultimodalBatch):
    model = MindGraphMultimodal()
    s = model.summary()
    assert s["parameters"] > 0
    bench = model.benchmark(batch, runs=3)
    assert "mean_ms" in bench


def test_device_transfer(batch: MultimodalBatch):
    device = resolve_device("cpu")
    dtype = resolve_dtype("fp32", device)
    model = MindGraphMultimodal().to(device)
    out = model.forward(batch.to(device))
    assert out["logits"].dtype == dtype or out["logits"].dtype == torch.float32


def test_missing_modality():
    b, d = 2, 768
    batch = MultimodalBatch(
        text=torch.randn(b, 8, d),
        text_mask=torch.ones(b, 8),
        modality_presence={"audio": torch.zeros(b), "text": torch.ones(b), "visual": torch.zeros(b), "image": torch.zeros(b)},
        labels=torch.tensor([0, 1]),
    )
    model = MindGraphMultimodal()
    out = model.forward(batch)
    assert out["logits"].shape == (2, 2)


def test_multimodal_batch_from_collated():
    collated = {
        "audio": torch.randn(2, 10),
        "text": torch.randn(2, 10),
        "labels": torch.tensor([0, 1]),
        "participant_ids": ["300", "301"],
        "modality_presence": {"audio": torch.ones(2), "text": torch.ones(2)},
    }
    batch = MultimodalBatch.from_collated(collated)
    assert batch.participant_ids == ["300", "301"]


def test_registry(tmp_path: Path):
    reg = ModelRegistry(path=tmp_path / "registry.json")
    assert reg.get("MODEL_010") is not None
    reg.save()
    assert (tmp_path / "registry.json").exists()


def test_explainability(batch: MultimodalBatch):
    model = MindGraphMultimodal()
    out = model.predict_with_explanation(batch)
    assert "explanation" in out


def test_torchscript_export(tmp_path: Path):
    model = MindGraphMultimodal()
    path = model.export(tmp_path / "model.pt", fmt="torchscript")
    if path is not None:
        loaded = torch.jit.load(str(path))
        assert loaded is not None
