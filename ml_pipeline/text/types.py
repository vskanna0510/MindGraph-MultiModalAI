"""Text pipeline domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Utterance:
    speaker: str
    text: str
    start_time: float | None = None
    stop_time: float | None = None
    order: int = 0


@dataclass
class LanguageInfo:
    primary: str
    secondary: str | None = None
    mixed: bool = False
    confidence: float = 0.0
    distribution: dict[str, float] = field(default_factory=dict)
    code_mix_tokens: dict[str, str] = field(default_factory=dict)


@dataclass
class PsycholinguisticFeatures:
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    vocabulary_size: int = 0
    lexical_diversity: float = 0.0
    pronoun_frequency: float = 0.0
    self_references: int = 0
    future_references: int = 0
    past_references: int = 0
    sleep_mentions: int = 0
    family_mentions: int = 0
    work_mentions: int = 0
    isolation_indicators: int = 0
    hopelessness_indicators: int = 0
    stress_indicators: int = 0
    fatigue_indicators: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextEmbedding:
    participant_id: str
    session_id: str
    model_name: str
    model_version: str
    embedding_dim: int
    vector_path: Path
    language: str
    tokenizer: str
    config_version: str
    extraction_timestamp: str
    text_hash: str = ""


@dataclass
class PipelineStageResult:
    stage: str
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class TextPipelineResult:
    participant_id: str
    source_path: Path
    raw_text: str = ""
    cleaned_text: str = ""
    utterances: list[Utterance] = field(default_factory=list)
    language: LanguageInfo | None = None
    sentences: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    psycholinguistic: PsycholinguisticFeatures | None = None
    sentiment: dict[str, float] = field(default_factory=dict)
    emotions: dict[str, float] = field(default_factory=dict)
    keywords: dict[str, list[str]] = field(default_factory=dict)
    embedding: TextEmbedding | None = None
    quality_score: float = 0.0
    stages: list[PipelineStageResult] = field(default_factory=list)
    validation_passed: bool = True
