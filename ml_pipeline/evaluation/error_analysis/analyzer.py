"""Error analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def analyze_errors(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray | None = None,
    participant_ids: list[str] | None = None,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    valid = labels >= 0
    preds, labels = preds[valid], labels[valid]
    if probs is not None:
        conf = probs[valid].max(axis=1) if probs.ndim == 2 else probs[valid]
    else:
        conf = np.ones(len(labels))

    fp_mask = (preds == 1) & (labels == 0)
    fn_mask = (preds == 0) & (labels == 1)
    low_conf = conf < 0.5
    high_conf_err = ((preds != labels) & (conf >= 0.8))

    result: dict[str, Any] = {
        "false_positives": int(fp_mask.sum()),
        "false_negatives": int(fn_mask.sum()),
        "low_confidence_count": int(low_conf.sum()),
        "high_confidence_errors": int(high_conf_err.sum()),
        "recommendations": [],
    }

    if fp_mask.sum() > fn_mask.sum():
        result["recommendations"].append("Elevated false positives — review threshold calibration and negative class balance.")
    if fn_mask.sum() > 0:
        result["recommendations"].append("False negatives present — consider recall-focused loss weighting.")
    if high_conf_err.sum() > 0:
        result["recommendations"].append("High-confidence errors detected — run calibration and error case review.")

    if participant_ids:
        pids = np.array(participant_ids)[valid]
        result["misclassified_participants"] = sorted(set(pids[preds != labels].tolist()))
    if languages:
        langs = np.array(languages)[valid]
        mis_lang: dict[str, int] = {}
        for lang in set(langs.tolist()):
            m = (preds != labels) & (langs == lang)
            mis_lang[str(lang)] = int(m.sum())
        result["misclassified_by_language"] = mis_lang

    return result
