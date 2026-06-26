"""Psycholinguistic feature extraction."""

from __future__ import annotations

import re

from ml_pipeline.text.features.keywords import DEFAULT_LEXICONS, extract_keywords
from ml_pipeline.text.types import PsycholinguisticFeatures

PRONOUN_RE = re.compile(r"\b(i|me|my|myself|mine)\b", re.I)
FUTURE_RE = re.compile(r"\b(will|gonna|going to|tomorrow|future)\b", re.I)
PAST_RE = re.compile(r"\b(was|were|had|yesterday|ago|used to)\b", re.I)
FAMILY_RE = re.compile(r"\b(mother|father|parent|family|sibling|brother|sister)\b", re.I)
WORK_RE = re.compile(r"\b(work|job|boss|office|career|employ)\b", re.I)


def extract_psycholinguistic(text: str, sentences: list[str], tokens: list[str]) -> PsycholinguisticFeatures:
    words = [t for t in tokens if t.strip()]
    vocab = set(w.lower() for w in words)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    kw = extract_keywords(text, DEFAULT_LEXICONS)

    return PsycholinguisticFeatures(
        word_count=word_count,
        sentence_count=len(sentences),
        avg_sentence_length=word_count / sentence_count,
        vocabulary_size=len(vocab),
        lexical_diversity=len(vocab) / max(word_count, 1),
        pronoun_frequency=len(PRONOUN_RE.findall(text)) / max(word_count, 1),
        self_references=len(PRONOUN_RE.findall(text)),
        future_references=len(FUTURE_RE.findall(text)),
        past_references=len(PAST_RE.findall(text)),
        sleep_mentions=len(kw.get("sleep", [])),
        family_mentions=len(FAMILY_RE.findall(text)),
        work_mentions=len(WORK_RE.findall(text)),
        isolation_indicators=len(kw.get("isolation", [])),
        hopelessness_indicators=len(kw.get("depression", [])),
        stress_indicators=len(kw.get("stress", [])),
        fatigue_indicators=len(kw.get("fatigue", [])),
        extra={"academic_mentions": len(re.findall(r"\b(school|college|university|study)\b", text, re.I))},
    )
