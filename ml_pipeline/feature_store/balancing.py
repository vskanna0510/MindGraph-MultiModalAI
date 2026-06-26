"""Class balancing utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def compute_class_distribution(labels: list[int | str]) -> dict[str, Any]:
    mapped = []
    for l in labels:
        if str(l).lower() in {"depression", "1", "depressed"}:
            mapped.append("depression")
        elif str(l).lower() in {"normal", "0"}:
            mapped.append("normal")
        else:
            mapped.append("unknown")
    counts = Counter(mapped)
    total = sum(counts.values()) or 1
    minority = min(counts.values()) if counts else 0
    majority = max(counts.values()) if counts else 0
    return {
        "distribution": dict(counts),
        "minority_ratio": minority / total,
        "majority_ratio": majority / total,
        "total": total,
    }


def class_weights(labels: list[int]) -> np.ndarray:
    valid = [l for l in labels if l >= 0]
    if not valid:
        return np.ones(2, dtype=np.float32)
    counts = Counter(valid)
    total = sum(counts.values())
    weights = {c: total / (len(counts) * cnt) for c, cnt in counts.items()}
    max_class = max(counts.keys()) + 1
    arr = np.ones(max_class, dtype=np.float32)
    for c, w in weights.items():
        arr[c] = w
    return arr
