"""Training metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_metrics(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray | None = None) -> dict[str, float]:
    valid = labels >= 0
    preds, labels = preds[valid], labels[valid]
    if len(labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    acc = float((preds == labels).mean())
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    sensitivity = recall

    metrics: dict[str, float] = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
    }

    if probs is not None and probs.shape[-1] >= 2:
        try:
            from sklearn.metrics import roc_auc_score

            p = probs[valid, 1] if probs.ndim == 2 else probs[valid]
            if len(np.unique(labels)) > 1:
                metrics["roc_auc"] = float(roc_auc_score(labels, p))
            else:
                metrics["roc_auc"] = 0.0
        except Exception:
            metrics["roc_auc"] = 0.0
        try:
            from sklearn.metrics import matthews_corrcoef, cohen_kappa_score

            metrics["mcc"] = float(matthews_corrcoef(labels, preds))
            metrics["cohen_kappa"] = float(cohen_kappa_score(labels, preds))
        except Exception:
            pass
    return metrics


def metrics_from_outputs(outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> dict[str, float]:
    preds = outputs.get("preds", outputs["logits"].argmax(dim=-1)).detach().cpu().numpy()
    probs = outputs.get("probs", torch.softmax(outputs["logits"], dim=-1)).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    return compute_metrics(preds, y, probs)
