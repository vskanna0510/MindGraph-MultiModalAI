"""Training framework unit tests (MP3 Part 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ml_pipeline.models.heads.multi_task import MultiTaskHeads
from ml_pipeline.models.losses.multimodal import MultimodalLoss
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.models.types import MultimodalBatch
from ml_pipeline.training.checkpoint import CheckpointManager
from ml_pipeline.training.early_stopping import EarlyStopping
from ml_pipeline.training.losses.classification import focal_loss, label_smoothing_ce
from ml_pipeline.training.losses.multitask import MultiTaskLoss, compute_class_weights
from ml_pipeline.training.losses.regression import build_regression_loss
from ml_pipeline.training.metrics import compute_metrics
from ml_pipeline.training.optimizers import build_optimizer
from ml_pipeline.training.schedulers import build_scheduler
from ml_pipeline.training.stages import apply_training_stage, stage_for_epoch
from ml_pipeline.training.trainer import Trainer


@pytest.fixture
def batch():
    b, t, d = 4, 8, 768
    return MultimodalBatch(
        audio=torch.randn(b, t, d),
        text=torch.randn(b, t, d),
        audio_mask=torch.ones(b, t),
        text_mask=torch.ones(b, t),
        modality_presence={"audio": torch.ones(b), "visual": torch.zeros(b), "text": torch.ones(b), "image": torch.zeros(b)},
        labels=torch.tensor([0, 1, 1, 0]),
    )


def test_multi_task_heads():
    heads = MultiTaskHeads(128, {"hidden_dim": 64, "num_classes": 2})
    out = heads(torch.randn(2, 128))
    assert out["logits"].shape == (2, 2)
    assert out["risk_score"].shape == (2,)
    assert out["confidence"].shape == (2, 5)
    assert out["temporal_risks"].shape == (2, 4)


def test_focal_loss():
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 1, 0])
    loss = focal_loss(logits, targets, gamma=2.0)
    assert loss.item() > 0


def test_label_smoothing():
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 1, 0])
    loss = label_smoothing_ce(logits, targets, smoothing=0.1)
    assert loss.item() > 0


def test_multitask_loss(batch):
    model = MindGraphMultimodal()
    out = model(batch)
    loss_fn = MultiTaskLoss({"loss": {"weights": {"classification": 1.0, "regression": 0.3}}})
    losses = loss_fn(out, batch.labels, model)
    assert losses["loss"].requires_grad
    losses["loss"].backward()


def test_multimodal_loss_compat(batch):
    model = MindGraphMultimodal()
    out = model(batch)
    loss_fn = MultimodalLoss()
    losses = loss_fn(out["logits"], batch.labels, out["risk_logits"])
    assert "loss" in losses


def test_class_weights():
    w = compute_class_weights(torch.tensor([0, 0, 1, 1, 1]))
    assert w.shape == (2,)


def test_regression_losses():
    huber = build_regression_loss("huber")
    pred = torch.tensor([0.2, 0.8])
    target = torch.tensor([0.0, 1.0])
    assert huber(pred, target).item() >= 0


def test_metrics():
    preds = torch.tensor([0, 1, 1, 0]).numpy()
    labels = torch.tensor([0, 1, 0, 0]).numpy()
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]]).numpy()
    m = compute_metrics(preds, labels, probs)
    assert "f1" in m
    assert "accuracy" in m


def test_optimizer_factory():
    model = nn.Linear(10, 2)
    opt = build_optimizer("adamw", model.parameters(), {"learning_rate": 1e-3, "weight_decay": 0.01})
    assert opt is not None


def test_scheduler_factory():
    model = nn.Linear(10, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler("cosine", opt, {"min_lr": 1e-6}, steps_per_epoch=10, total_epochs=20)
    assert sched is not None


def test_early_stopping():
    es = EarlyStopping(patience=2, monitor="val_f1", mode="max")
    assert not es.step({"val_f1": 0.5})
    assert not es.step({"val_f1": 0.6})
    assert not es.step({"val_f1": 0.55})
    assert es.step({"val_f1": 0.54})


def test_checkpoint_manager(tmp_path: Path):
    model = nn.Linear(4, 2)
    ckpt = CheckpointManager(tmp_path, save_top_k=2, monitors={"val_f1": "max"})
    path = ckpt.save(model, 0, {"val_f1": 0.7}, tag="latest")
    assert path.exists()
    ckpt.maybe_save_best(model, 1, {"val_f1": 0.8})


def test_training_stages():
    model = MindGraphMultimodal()
    apply_training_stage(model, "encoder_warmup")
    assert any(p.requires_grad for p in model.audio_encoder.parameters())
    apply_training_stage(model, "end_to_end")
    assert all(p.requires_grad for p in model.parameters())


def test_stage_for_epoch():
    cfg = {"encoder_warmup_epochs": 2, "fusion_epochs": 3, "temporal_epochs": 0, "graph_epochs": 0}
    assert stage_for_epoch(0, cfg) == "encoder_warmup"
    assert stage_for_epoch(3, cfg) == "fusion"
    assert stage_for_epoch(10, cfg) == "end_to_end"


def test_trainer_dry_run(batch):
    model = MindGraphMultimodal()
    cfg = {"training": {"max_epochs": 1, "mixed_precision": False}, "optimization": {}, "checkpoint": {}, "logging": {}, "stages": {}}
    trainer = Trainer(model, cfg, train_loader=None, val_loader=None)
    m = trainer.train_epoch(0)
    assert "train_loss" in m


def test_mindgraph_multi_task_outputs(batch):
    model = MindGraphMultimodal()
    out = model(batch)
    assert "risk_score" in out
    assert "confidence" in out
    assert "temporal_risks" in out
