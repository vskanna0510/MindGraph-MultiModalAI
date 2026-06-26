"""Tests for independent modality encoders (MP3 Part 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml_pipeline.models.audio.base_audio_encoder import BaseAudioEncoder
from ml_pipeline.models.audio.feature_fusion import AudioFeatureFusion
from ml_pipeline.models.audio.hubert_encoder import HuBERTEncoder
from ml_pipeline.models.audio.wav2vec2_encoder import Wav2Vec2Encoder
from ml_pipeline.models.audio.wavlm_encoder import WavLMEncoder
from ml_pipeline.models.encoders.benchmark import benchmark_encoder
from ml_pipeline.models.encoders.output import EncoderOutput
from ml_pipeline.models.encoders.pooling import AttentionPooling
from ml_pipeline.models.encoders.projection_head import ProjectionHead
from ml_pipeline.models.image.clip_encoder import CLIPEncoder
from ml_pipeline.models.image.vit_image_encoder import ViTImageEncoder
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.models.text.indicbert_encoder import IndicBERTEncoder
from ml_pipeline.models.text.mbert_encoder import MBERTEncoder
from ml_pipeline.models.text.xlmr_encoder import XLMRoBERTaEncoder
from ml_pipeline.models.visual.mediapipe_encoder import MediaPipeEncoder
from ml_pipeline.models.visual.vit_encoder import ViTEncoder


CFG = {"input_dim": 512, "hidden_dim": 256, "latent_dim": 256, "num_layers": 2, "dropout": 0.1, "pooling": "attention"}


def test_encoder_output_contract():
    out = EncoderOutput(embedding=torch.randn(2, 256), confidence=torch.ones(2))
    assert out.pooled.shape == (2, 256)


def test_projection_head():
    head = ProjectionHead(512, 256)
    x = head(torch.randn(2, 10, 512))
    assert x.shape == (2, 10, 256)


def test_attention_pooling_modes():
    hidden = torch.randn(2, 8, 256)
    mask = torch.ones(2, 8)
    for mode in ("mean", "max", "attention", "cls"):
        pool = AttentionPooling(256, mode)
        pooled, attn = pool(hidden, mask)
        assert pooled.shape == (2, 256)
        assert attn.shape[0] == 2


def test_audio_feature_fusion():
    fusion = AudioFeatureFusion({"mfcc": 40, "prosody": 32, "wav2vec": 256}, 256)
    features = {
        "mfcc": torch.randn(2, 10, 40),
        "prosody": torch.randn(2, 10, 32),
        "wav2vec": torch.randn(2, 10, 256),
    }
    fused, attn = fusion(features)
    assert fused.shape == (2, 10, 256)
    assert attn.shape[0] == 2


@pytest.mark.parametrize("cls", [Wav2Vec2Encoder, HuBERTEncoder, WavLMEncoder, BaseAudioEncoder])
def test_audio_encoders_forward(cls):
    enc = cls(CFG)
    out = enc(torch.randn(2, 16, 512), torch.ones(2, 16))
    assert isinstance(out, EncoderOutput)
    assert out.embedding.shape == (2, 256)
    assert out.attention_maps is not None
    assert out.confidence is not None


def test_audio_encoder_backward():
    enc = Wav2Vec2Encoder(CFG)
    out = enc(torch.randn(2, 16, 512), torch.ones(2, 16))
    out.embedding.sum().backward()
    assert any(p.grad is not None for p in enc.parameters())


def test_audio_encoder_save_load(tmp_path: Path):
    enc = Wav2Vec2Encoder(CFG)
    path = tmp_path / "audio.pt"
    enc.save(path)
    loaded = Wav2Vec2Encoder.load_checkpoint(path, Wav2Vec2Encoder)
    assert loaded.config == enc.config


@pytest.mark.parametrize("cls", [ViTEncoder, MediaPipeEncoder])
def test_visual_encoders(cls):
    enc = cls(CFG)
    out = enc(torch.randn(2, 8, 512), torch.ones(2, 8))
    assert out.embedding.shape == (2, 256)


@pytest.mark.parametrize("cls", [IndicBERTEncoder, MBERTEncoder, XLMRoBERTaEncoder])
def test_text_encoders(cls):
    enc = cls(CFG)
    out = enc(torch.randn(2, 12, 512), torch.ones(2, 12))
    assert out.embedding.shape == (2, 256)
    assert "languages" in out.metadata


@pytest.mark.parametrize("cls", [ViTImageEncoder, CLIPEncoder])
def test_image_encoders(cls):
    enc = cls(CFG)
    out = enc(torch.randn(2, 8, 512))
    assert out.embedding.shape == (2, 256)


def test_encoder_benchmark():
    enc = Wav2Vec2Encoder(CFG)
    stats = benchmark_encoder(enc, torch.randn(1, 16, 512), torch.ones(1, 16), runs=3)
    assert stats["mean_latency_ms"] > 0
    assert stats["parameters"] > 0


def test_mindgraph_with_new_encoders():
    model = MindGraphMultimodal()
    batch = model._dummy_batch()
    out = model.forward(batch)
    assert out["logits"].shape[0] == 1


def test_variable_sequence_length():
    enc = IndicBERTEncoder(CFG)
    for t in (4, 16, 32):
        out = enc(torch.randn(2, t, 512), torch.ones(2, t))
        assert out.embedding.shape == (2, 256)


def test_torchscript_encoder_export(tmp_path: Path):
    enc = BaseAudioEncoder(CFG)
    path = enc.export(tmp_path / "audio.ts", "torchscript")
    if path is not None:
        assert path.exists()
