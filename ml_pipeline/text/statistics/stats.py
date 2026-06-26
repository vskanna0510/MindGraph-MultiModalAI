"""Pipeline statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any


def compute_statistics(results: list[Any]) -> dict[str, Any]:
    langs = Counter()
    sentiments = Counter()
    emotions = Counter()
    tokens_total = 0
    quality_scores: list[float] = []

    for r in results:
        if r.language:
            langs[r.language.primary] += 1
        if r.sentiment.get("label"):
            sentiments[str(r.sentiment["label"])] += 1
        if r.emotions:
            top = max(r.emotions, key=r.emotions.get)
            emotions[top] += 1
        tokens_total += len(r.tokens)
        quality_scores.append(r.quality_score)

    return {
        "sample_count": len(results),
        "language_distribution": dict(langs),
        "avg_tokens": tokens_total / max(len(results), 1),
        "sentiment_distribution": dict(sentiments),
        "emotion_distribution": dict(emotions),
        "avg_quality_score": sum(quality_scores) / max(len(quality_scores), 1),
    }
