"""Tests for text engineering pipeline (MP2 Part 5)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from ml_pipeline.text.augmentation.augment import augment_text
from ml_pipeline.text.cache.manager import TextCacheManager
from ml_pipeline.text.embeddings.extractor import EmbeddingExtractor
from ml_pipeline.text.export.exporter import export_features
from ml_pipeline.text.features.keywords import extract_keywords, load_lexicons
from ml_pipeline.text.features.psycholinguistic import extract_psycholinguistic
from ml_pipeline.text.language.detection import detect_code_mix, detect_language
from ml_pipeline.text.loader.transcript import load_transcript
from ml_pipeline.text.loaders.dataloader import collate_text_batch
from ml_pipeline.text.loaders.datasets import DAICTextDataset
from ml_pipeline.text.nlp.analysis import analyze_sentiment, detect_emotions, extract_pos_tags
from ml_pipeline.text.pipeline import TextPipeline
from ml_pipeline.text.preprocess.cleaning import clean_text, validate_encoding
from ml_pipeline.text.quality.validation import assess_quality, validate_result
from ml_pipeline.text.segmentation.sentences import segment_sentences
from ml_pipeline.text.tokenization.tokenizer import TextTokenizer


@pytest.fixture
def sample_transcript(tmp_path: Path) -> Path:
    path = tmp_path / "300_TRANSCRIPT.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["start_time", "stop_time", "speaker", "value"])
        writer.writerow(["1.0", "2.0", "Participant", "I feel sad and lonely sometimes"])
        writer.writerow(["2.1", "3.0", "Participant", "but I am trying to get better"])
    return path


@pytest.fixture
def daic_text_layout(tmp_path: Path, sample_transcript: Path) -> Path:
    root = tmp_path / "daic_woz" / "300_P"
    root.mkdir(parents=True)
    dest = root / "300_TRANSCRIPT.csv"
    dest.write_text(sample_transcript.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path / "daic_woz"


def test_validate_encoding():
    ok, errs = validate_encoding("hello world")
    assert ok
    assert not errs


def test_clean_text_preserves_negation():
    text = "I am not happy"
    cleaned = clean_text(text)
    assert "not" in cleaned


def test_load_transcript(sample_transcript: Path):
    text, utterances = load_transcript(sample_transcript)
    assert "sad" in text
    assert len(utterances) == 2
    assert utterances[0].speaker == "Participant"


def test_language_detection():
    info = detect_language("I feel anxious and stressed")
    assert info.primary in {"en", "mixed"}
    assert info.confidence > 0


def test_code_mix_detection():
    tags = detect_code_mix("hello world")
    assert tags["hello"] == "en"


def test_segment_sentences():
    sents = segment_sentences("First sentence. Second one!")
    assert len(sents) == 2


def test_tokenizer_whitespace():
    tok = TextTokenizer(method="whitespace")
    tokens = tok.tokenize("hello world test")
    assert tokens == ["hello", "world", "test"]


def test_sentiment_and_emotion():
    sent = analyze_sentiment("I am happy and good")
    assert sent["label"] in {"positive", "neutral"}
    emo = detect_emotions("I feel hopeless and lonely")
    assert emo["hopelessness"] > 0 or emo["loneliness"] > 0


def test_psycholinguistic_and_keywords():
    text = "I feel alone and tired. I cannot sleep."
    tokens = text.split()
    feats = extract_psycholinguistic(text, segment_sentences(text), tokens)
    assert feats.self_references >= 1
    kw = extract_keywords(text, load_lexicons())
    assert "isolation" in kw or "sleep" in kw


def test_pos_tags():
    pos = extract_pos_tags(["I", "am", "not", "happy"])
    assert "not" in pos["negations"]


def test_embedding_extractor(tmp_path: Path):
    ext = EmbeddingExtractor(model_name="bert-base-uncased")
    vec = ext.extract("sample text for embedding")
    assert vec.shape[-1] == 768
    emb = ext.save_embedding(vec, tmp_path, "300", "300", "en", "sample")
    assert emb.vector_path.exists()


def test_cache_manager(tmp_path: Path):
    cache = TextCacheManager(tmp_path, "cfg1")
    cache.set("tokenization", "300", {"tokens": ["a", "b"]})
    hit = cache.get("tokenization", "300")
    assert hit["tokens"] == ["a", "b"]


def test_quality_and_validation(sample_transcript: Path):
    pipeline = TextPipeline()
    result = pipeline.process_file(sample_transcript, "300", "300")
    assert result.quality_score > 0
    ok, errs = validate_result(result)
    assert ok
    assert not errs


def test_text_pipeline_end_to_end(sample_transcript: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "ml_pipeline.text.pipeline.text_paths",
        lambda: {
            "processed": tmp_path / "processed",
            "reports": tmp_path / "reports",
            "cache": tmp_path / "cache",
        },
    )
    pipeline = TextPipeline()
    result = pipeline.process_file(sample_transcript, "300", "300")
    assert result.validation_passed
    assert result.embedding is not None
    assert len(result.stages) >= 15
    exports = pipeline.export_all([result])
    assert "csv" in exports or "json" in exports


def test_daic_dataset_discover(daic_text_layout: Path):
    ds = DAICTextDataset(raw_root=daic_text_layout)
    records = ds.discover()
    assert len(records) == 1
    assert records[0]["participant_id"] == "300"


def test_export_features(tmp_path: Path):
    rows = [{"participant_id": "300", "language": "en", "quality_score": 0.9}]
    paths = export_features(rows, tmp_path, ["csv", "json", "parquet"])
    assert paths["csv"].exists()
    assert paths["json"].exists()
    assert "parquet" in paths or "csv" in paths


def test_collate_batch():
    batch = [
        {"participant_id": "1", "tokens": ["a"], "length": 1, "sentiment": {}, "label": 0},
        {"participant_id": "2", "tokens": ["a", "b"], "length": 2, "sentiment": {}, "label": 1},
    ]
    out = collate_text_batch(batch)
    assert out["lengths"].tolist() == [1, 2]
    assert out["attention_mask"].shape == (2, 2)


def test_augment_text():
    text = "one two three four five"
    out = augment_text(text, seed=1, synonym_prob=0.5)
    assert isinstance(out, str)


def test_assess_quality():
    score = assess_quality("hello world " * 10, language_confidence=0.9)
    assert 0 < score <= 1.0
