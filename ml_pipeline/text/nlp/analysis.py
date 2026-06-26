"""Sentiment and emotion analysis."""

from __future__ import annotations

import re
from typing import Any

POSITIVE = {"good", "happy", "great", "love", "fine", "better", "wonderful"}
NEGATIVE = {"bad", "sad", "terrible", "awful", "hate", "worse", "depressed", "lonely", "anxious"}


def analyze_sentiment(text: str) -> dict[str, float]:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(tokens & POSITIVE)
    neg = len(tokens & NEGATIVE)
    total = pos + neg + 1
    compound = (pos - neg) / total
    if compound > 0.1:
        label = "positive"
    elif compound < -0.1:
        label = "negative"
    else:
        label = "neutral"
    return {
        "positive": pos / total,
        "negative": neg / total,
        "neutral": 1.0 - abs(compound),
        "compound": compound,
        "label": label,
        "confidence": min(1.0, abs(compound) + 0.5),
        "intensity": abs(compound),
    }


def detect_emotions(text: str, labels: list[str] | None = None) -> dict[str, float]:
    labels = labels or [
        "joy", "sadness", "fear", "anger", "disgust", "surprise",
        "neutral", "hopelessness", "loneliness", "anxiety", "stress",
    ]
    lower = text.lower()
    mapping = {
        "joy": ["happy", "joy", "glad", "love"],
        "sadness": ["sad", "depressed", "cry", "down"],
        "fear": ["afraid", "scared", "fear", "panic"],
        "anger": ["angry", "mad", "furious"],
        "disgust": ["disgust", "gross"],
        "surprise": ["surprise", "shocked"],
        "hopelessness": ["hopeless", "worthless", "pointless"],
        "loneliness": ["lonely", "alone", "isolated"],
        "anxiety": ["anxious", "worried", "nervous"],
        "stress": ["stressed", "overwhelmed"],
    }
    scores: dict[str, float] = {}
    for label in labels:
        terms = mapping.get(label, [])
        hits = sum(1 for t in terms if t in lower)
        scores[label] = hits / max(len(terms), 1)
    if max(scores.values()) < 0.01:
        scores["neutral"] = 1.0
    total = sum(scores.values()) or 1.0
    return {k: round(v / total, 4) for k, v in scores.items()}


def extract_pos_tags(tokens: list[str]) -> dict[str, list[str]]:
    """Lightweight POS buckets for tests and fallback."""
    nouns, verbs, pronouns, adjectives, adverbs, negations = [], [], [], [], [], []
    neg_words = {"not", "no", "never", "n't", "cannot", "can't"}
    for tok in tokens:
        low = tok.lower()
        if low in neg_words:
            negations.append(tok)
        elif low in {"i", "you", "he", "she", "we", "they", "me", "my"}:
            pronouns.append(tok)
        elif low.endswith("ly"):
            adverbs.append(tok)
        elif low.endswith(("ed", "ing")):
            verbs.append(tok)
        else:
            nouns.append(tok)
    return {
        "nouns": nouns,
        "verbs": verbs,
        "pronouns": pronouns,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "negations": negations,
    }


def extract_ner(text: str) -> dict[str, list[str]]:
    """Rule-based NER fallback."""
    import re

    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    times = re.findall(r"\b\d{1,2}:\d{2}\b", text)
    places = re.findall(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b", text)
    return {
        "person": [],
        "place": places[:5],
        "organization": [],
        "date": dates,
        "time": times,
        "medical": [],
        "mental_health": [],
    }
