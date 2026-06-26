"""Training-only text augmentation."""

from __future__ import annotations

import random


def augment_text(text: str, seed: int = 42, synonym_prob: float = 0.0) -> str:
    """Controlled augmentation; never changes labels externally."""
    rng = random.Random(seed)
    tokens = text.split()
    if synonym_prob <= 0 or len(tokens) < 3:
        return text
    out = []
    for tok in tokens:
        if rng.random() < synonym_prob:
            continue
        out.append(tok)
    return " ".join(out) if out else text
