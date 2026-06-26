"""Classification metrics suite."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def classification_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    valid = labels >= 0
    preds, labels = preds[valid], labels[valid]
    if len(labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0}

    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = float((preds == labels).mean())
    tpr = recall
    tnr = specificity
    balanced_acc = (tpr + tnr) / 2
    ber = 1 - balanced_acc

    metrics: dict[str, float] = {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_error_rate": ber,
        "support": float(len(labels)),
    }

    try:
        from sklearn.metrics import (
            average_precision_score,
            cohen_kappa_score,
            f1_score,
            log_loss,
            matthews_corrcoef,
            roc_auc_score,
        )

        metrics["weighted_f1"] = float(f1_score(labels, preds, average="weighted", zero_division=0))
        metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
        metrics["micro_f1"] = float(f1_score(labels, preds, average="micro", zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(labels, preds))
        metrics["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
        if probs is not None and probs.ndim == 2 and probs.shape[1] >= 2:
            p = probs[valid, 1]
            if len(np.unique(labels)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(labels, p))
                metrics["pr_auc"] = float(average_precision_score(labels, p))
                metrics["average_precision"] = metrics["pr_auc"]
            metrics["log_loss"] = float(log_loss(labels, probs[valid], labels=[0, 1]))
            metrics["brier_score"] = float(np.mean((p - labels) ** 2))
    except Exception:
        pass
    return metrics


def per_class_report(preds: np.ndarray, labels: np.ndarray, class_names: list[str] | None = None) -> dict[str, dict[str, float]]:
    valid = labels >= 0
    preds, labels = preds[valid], labels[valid]
    classes = sorted(set(labels.tolist()))
    if class_names is None:
        class_names = [str(c) for c in classes]
    report: dict[str, dict[str, float]] = {}
    for c, name in zip(classes, class_names):
        mask = labels == c
        pred_c = preds == c
        tp = float((pred_c & mask).sum())
        fp = float((pred_c & ~mask).sum())
        fn = float((~pred_c & mask).sum())
        tn = float((~pred_c & ~mask).sum())
        report[name] = {
            "precision": _safe_div(tp, tp + fp),
            "recall": _safe_div(tp, tp + fn),
            "specificity": _safe_div(tn, tn + fp),
            "support": float(mask.sum()),
        }
    return report
