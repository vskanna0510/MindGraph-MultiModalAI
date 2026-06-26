"""Sentence and utterance segmentation."""

from __future__ import annotations

import re

from ml_pipeline.text.types import Utterance

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def segment_sentences(text: str) -> list[str]:
  sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
  return sentences if sentences else ([text] if text.strip() else [])


def segment_utterances(utterances: list[Utterance]) -> list[str]:
  return [u.text for u in sorted(utterances, key=lambda x: x.order) if u.text.strip()]
