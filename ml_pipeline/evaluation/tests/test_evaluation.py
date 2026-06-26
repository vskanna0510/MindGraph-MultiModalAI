"""Evaluation framework unit tests (MP3 Part 6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml_pipeline.evaluation.ablation.variants import apply_ablation_mask, list_ablation_variants
from ml_pipeline.evaluation.benchmark.compare import benchmark_inference
from ml_pipeline.evaluation.calibration.ece import expected_calibration_error, TemperatureScaler
from ml_pipeline.evaluation.confusion.matrix import confusion_matrix_data
from ml_pipeline.evaluation.error_analysis.analyzer import analyze_errors
from ml_pipeline.evaluation.evaluator import ModelEvaluator
from ml_pipeline.evaluation.explainability.xai import modality_contribution
from ml_pipeline.evaluation.metrics.classification import classification_metrics, per_class_report
from ml_pipeline.evaluation.metrics.regression import regression_metrics
from ml_pipeline.evaluation.pr.curves import pr_curve_data
from ml_pipeline.evaluation.roc.curves import roc_curve_data
from ml_pipeline.evaluation.statistics import confidence_interval, descriptive_stats, paired_t_test
from ml_pipeline.evaluation.uncertainty.estimation import predictive_entropy
from ml_pipeline.models.multimodal import MindGraphMultimodal
from ml_pipeline.models.types import MultimodalBatch


@pytest.fixture
def synthetic():
    n = 100
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, n)
    probs = rng.random((n, 2))
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return preds, labels, probs


def test_classification_metrics(synthetic):
    preds, labels, probs = synthetic
    m = classification_metrics(preds, labels, probs)
    assert "f1" in m
    assert "roc_auc" in m
    assert "balanced_accuracy" in m
    assert "log_loss" in m


def test_regression_metrics():
    pred = np.array([0.2, 0.5, 0.8])
    target = np.array([0.0, 0.5, 1.0])
    m = regression_metrics(pred, target)
    assert m["mae"] >= 0
    assert "r2" in m


def test_calibration_ece(synthetic):
    _, labels, probs = synthetic
    cal = expected_calibration_error(probs, labels)
    assert 0 <= cal.ece <= 1
    assert len(cal.bins) > 0


def test_temperature_scaling():
    logits = torch.randn(20, 2)
    labels = torch.randint(0, 2, (20,))
    scaler = TemperatureScaler()
    t = scaler.fit(logits, labels)
    assert t > 0


def test_confusion_matrix(synthetic):
    preds, labels, _ = synthetic
    cm = confusion_matrix_data(preds, labels)
    assert "matrix" in cm
    assert len(cm["classes"]) >= 2


def test_roc_pr(synthetic):
    _, labels, probs = synthetic
    roc = roc_curve_data(labels, probs[:, 1])
    pr = pr_curve_data(labels, probs[:, 1])
    assert "auc" in roc
    assert "average_precision" in pr


def test_per_class_report(synthetic):
    preds, labels, _ = synthetic
    report = per_class_report(preds, labels, ["neg", "pos"])
    assert "neg" in report or "0" in report or len(report) >= 1


def test_error_analysis(synthetic):
    preds, labels, probs = synthetic
    err = analyze_errors(preds, labels, probs)
    assert "false_positives" in err
    assert "recommendations" in err


def test_statistics():
    vals = [0.7, 0.75, 0.8, 0.72, 0.78]
    stats = descriptive_stats(vals)
    assert stats.mean > 0
    lo, hi = confidence_interval(vals)
    assert lo <= stats.mean <= hi


def test_paired_t_test():
    before = [0.7, 0.72, 0.68]
    after = [0.8, 0.82, 0.79]
    result = paired_t_test(before, after)
    assert result.test == "paired_t_test"


def test_modality_contribution():
    w = {"audio": torch.tensor([0.6, 0.4]), "text": torch.tensor([0.4, 0.6])}
    c = modality_contribution(w)
    assert abs(sum(c.values()) - 1.0) < 1e-5


def test_ablation_variants():
    assert "full_architecture" in list_ablation_variants()
    batch = MultimodalBatch(
        audio=torch.randn(2, 8, 64),
        modality_presence={"audio": torch.ones(2), "text": torch.zeros(2), "visual": torch.zeros(2), "image": torch.zeros(2)},
        labels=torch.tensor([0, 1]),
    )
    ablated = apply_ablation_mask(batch, "text_only")
    assert ablated.modality_presence["audio"].sum() == 0


def test_predictive_entropy():
    probs = np.array([0.7, 0.3])
    assert predictive_entropy(probs) > 0


def test_evaluator_explain():
    model = MindGraphMultimodal()
    batch = MultimodalBatch(
        audio=torch.randn(2, 8, 768),
        text=torch.randn(2, 8, 768),
        audio_mask=torch.ones(2, 8),
        text_mask=torch.ones(2, 8),
        modality_presence={"audio": torch.ones(2), "text": torch.ones(2), "visual": torch.zeros(2), "image": torch.zeros(2)},
        labels=torch.tensor([0, 1]),
    )
    ev = ModelEvaluator(model)
    expl = ev.explain_batch(batch)
    assert "modality_contribution" in expl


def test_benchmark_inference():
    model = MindGraphMultimodal()
    batch = model._dummy_batch()
    stats = benchmark_inference(model, batch, runs=2, warmup=1)
    assert "inference_time_ms" in stats
