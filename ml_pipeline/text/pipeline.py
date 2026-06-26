"""Text engineering pipeline orchestrator — 19 stages in fixed order."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_pipeline.text.augmentation.augment import augment_text
from ml_pipeline.text.cache.manager import TextCacheManager
from ml_pipeline.text.config import text_config, text_paths
from ml_pipeline.text.embeddings.extractor import EmbeddingExtractor
from ml_pipeline.text.export.exporter import export_features
from ml_pipeline.text.features.keywords import extract_keywords, load_lexicons
from ml_pipeline.text.features.psycholinguistic import extract_psycholinguistic
from ml_pipeline.text.language.detection import detect_code_mix, detect_language
from ml_pipeline.text.loader.transcript import load_transcript
from ml_pipeline.text.nlp.analysis import analyze_sentiment, detect_emotions, extract_ner, extract_pos_tags
from ml_pipeline.text.preprocess.cleaning import clean_text, normalize_text, validate_encoding
from ml_pipeline.text.quality.validation import assess_quality, validate_result
from ml_pipeline.text.segmentation.sentences import segment_sentences, segment_utterances
from ml_pipeline.text.statistics.stats import compute_statistics
from ml_pipeline.text.tokenization.tokenizer import TextTokenizer
from ml_pipeline.text.types import PipelineStageResult, TextPipelineResult
from ml_pipeline.text.visualization.plots import generate_visualizations
from ml_pipeline.utils.reproducibility import SeedBundle, set_global_seeds


class TextPipeline:
    """Production multilingual text processing pipeline (MP2 Part 5)."""

    def __init__(self) -> None:
        self.cfg = text_config()
        seeds = SeedBundle(python=int(self.cfg.get("pipeline", {}).get("seed", 42)))
        set_global_seeds(seeds, deterministic=self.cfg.get("pipeline", {}).get("deterministic", True))
        self.paths = text_paths()
        cfg_hash = hashlib.sha256(json.dumps(self.cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.cache = TextCacheManager(self.paths["cache"] / "text", cfg_hash)
        emb_cfg = self.cfg.get("embeddings", {})
        self.embedder = EmbeddingExtractor(
            model_name=emb_cfg.get("primary", "ai4bharat/indic-bert"),
            model_version=emb_cfg.get("model_version", "1.0"),
            device=emb_cfg.get("device", "auto"),
            config_version=self.cfg.get("pipeline", {}).get("version", "1.0.0"),
        )
        tok_cfg = self.cfg.get("tokenization", {})
        self.tokenizer = TextTokenizer(tok_cfg.get("method", "whitespace"), tok_cfg.get("model"))
        self.lexicons = load_lexicons(Path(self.cfg.get("keywords", {}).get("lexicon_dir", "configs/lexicons")))

    def process_file(
        self,
        source_path: Path,
        participant_id: str,
        session_id: str | None = None,
        split: str | None = None,
    ) -> TextPipelineResult:
        """Run all text pipeline stages on a single transcript."""
        session_id = session_id or participant_id
        result = TextPipelineResult(participant_id=participant_id, source_path=source_path)
        stages: list[PipelineStageResult] = []

        # 1 Load transcript
        raw_text, utterances = load_transcript(source_path)
        result.raw_text = raw_text
        result.utterances = utterances
        stages.append(PipelineStageResult("load", bool(raw_text.strip()), metadata={"chars": len(raw_text)}))

        # 2 Encoding validation
        enc_ok, enc_errs = validate_encoding(raw_text)
        stages.append(PipelineStageResult("encoding_validation", enc_ok, errors=enc_errs))

        # 3 Cleaning
        cleaned = clean_text(raw_text, preserve_clinical=True)
        result.cleaned_text = cleaned
        stages.append(PipelineStageResult("cleaning", bool(cleaned.strip())))

        # 4 Normalization
        text_cfg = self.cfg.get("text", {})
        normalized = normalize_text(cleaned, lowercase=bool(text_cfg.get("lowercase", False)))
        stages.append(PipelineStageResult("normalization", True))

        # Training-only augmentation
        if split == "train" and self.cfg.get("augmentation", {}).get("enabled", False):
            normalized = augment_text(
                normalized,
                seed=int(self.cfg.get("pipeline", {}).get("seed", 42)),
                synonym_prob=float(self.cfg.get("augmentation", {}).get("synonym_prob", 0.0)),
            )

        # 5 Language detection
        lang = detect_language(normalized)
        result.language = lang
        stages.append(PipelineStageResult("language_detection", True, metadata={"primary": lang.primary}))

        # 6 Code-mix detection
        code_mix = detect_code_mix(normalized)
        lang.code_mix_tokens = code_mix
        stages.append(PipelineStageResult("code_mix_detection", True, metadata={"tokens": len(code_mix)}))

        # 7 Sentence segmentation
        result.sentences = segment_sentences(normalized)
        utterance_texts = segment_utterances(utterances)
        if utterance_texts:
            result.sentences = utterance_texts
        stages.append(PipelineStageResult("sentence_segmentation", True, metadata={"sentences": len(result.sentences)}))

        # 8 Tokenization
        result.tokens = self.tokenizer.tokenize(normalized)
        stages.append(PipelineStageResult("tokenization", True, metadata={"tokens": len(result.tokens)}))

        # 9 POS tagging
        pos = extract_pos_tags(result.tokens)
        stages.append(PipelineStageResult("pos_tagging", True, metadata={"nouns": len(pos["nouns"])}))

        # 10 Dependency parsing (stub metadata)
        stages.append(PipelineStageResult("dependency_parsing", True, metadata={"root": result.tokens[0] if result.tokens else ""}))

        # 11 NER
        ner = extract_ner(normalized)
        stages.append(PipelineStageResult("ner", True, metadata={"entities": sum(len(v) for v in ner.values())}))

        # 12 Sentiment
        result.sentiment = analyze_sentiment(normalized)
        stages.append(PipelineStageResult("sentiment", True, metadata=result.sentiment))

        # 13 Emotion
        emotion_labels = self.cfg.get("emotion", {}).get("labels")
        result.emotions = detect_emotions(normalized, emotion_labels)
        stages.append(PipelineStageResult("emotion", True))

        # 14 Psycholinguistic features
        result.psycholinguistic = extract_psycholinguistic(normalized, result.sentences, result.tokens)
        stages.append(PipelineStageResult("psycholinguistic", True))

        # 15 Keywords
        result.keywords = extract_keywords(normalized, self.lexicons)
        stages.append(PipelineStageResult("keywords", True, metadata={"categories": list(result.keywords.keys())}))

        # Cache check for embeddings
        cache_key = f"{participant_id}:{session_id}"
        cached = self.cache.get("embeddings", cache_key)
        max_len = int(text_cfg.get("max_sequence_length", 512))

        if cached and Path(cached.get("vector_path", "")).exists():
            from ml_pipeline.text.types import TextEmbedding

            emb_meta = {**cached, "vector_path": Path(cached["vector_path"])}
            result.embedding = TextEmbedding(**emb_meta)
            stages.append(PipelineStageResult("caching", True, metadata={"hit": True}))
        else:
            # 16 Transformer embeddings
            vector = self.embedder.extract(normalized, max_length=max_len)
            emb_dir = self.paths["processed"] / "embeddings"
            result.embedding = self.embedder.save_embedding(
                vector, emb_dir, participant_id, session_id, lang.primary, normalized
            )
            emb_payload = {**result.embedding.__dict__, "vector_path": str(result.embedding.vector_path)}
            self.cache.set("embeddings", cache_key, emb_payload)
            stages.append(PipelineStageResult("embeddings", True, metadata={"dim": result.embedding.embedding_dim}))
            stages.append(PipelineStageResult("caching", True, metadata={"hit": False}))

        # 17 Embedding normalization metadata
        stages.append(PipelineStageResult("embedding_normalization", True, metadata={"max_length": max_len}))

        # 18 Quality
        unknown_ratio = 0.0
        result.quality_score = assess_quality(normalized, lang.confidence, unknown_ratio)
        stages.append(PipelineStageResult("quality", result.quality_score >= 0.4))

        # 19 Validation
        ok, val_errs = validate_result(result)
        result.validation_passed = ok
        stages.append(PipelineStageResult("validation", ok, errors=val_errs))

        result.stages = stages
        return result

    def process_batch(self, records: list[dict]) -> list[TextPipelineResult]:
        mp_cfg = self.cfg.get("multiprocessing", {})
        workers = int(mp_cfg.get("max_workers", 0))
        if mp_cfg.get("enabled", True) and workers != 1 and len(records) > 1:
            workers = workers or min(4, (os.cpu_count() or 2))
            results: list[TextPipelineResult] = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _process_worker,
                        str(r["file_path"]),
                        r["participant_id"],
                        r.get("session_id"),
                        r.get("split"),
                    ): r
                    for r in records
                }
                for fut in as_completed(futures):
                    results.append(fut.result())
            return results

        return [
            self.process_file(
                Path(r["file_path"]),
                r["participant_id"],
                r.get("session_id"),
                r.get("split"),
            )
            for r in records
        ]

    def save_metadata_parquet(self, results: list[TextPipelineResult]) -> Path:
        rows = []
        for r in results:
            rows.append(
                {
                    "participant_id": r.participant_id,
                    "source_path": str(r.source_path),
                    "language": r.language.primary if r.language else "",
                    "language_confidence": r.language.confidence if r.language else 0.0,
                    "word_count": r.psycholinguistic.word_count if r.psycholinguistic else 0,
                    "sentiment_label": r.sentiment.get("label", ""),
                    "quality_score": r.quality_score,
                    "embedding_path": str(r.embedding.vector_path) if r.embedding else "",
                    "validation_passed": r.validation_passed,
                }
            )
        out = self.paths["processed"] / self.cfg.get("paths", {}).get("metadata_file", "text_metadata.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(out, index=False)
        except ImportError:
            out = out.with_suffix(".csv")
            df.to_csv(out, index=False)
        return out

    def export_all(self, results: list[TextPipelineResult]) -> dict[str, Path]:
        rows = []
        for r in results:
            rows.append(
                {
                    "participant_id": r.participant_id,
                    "language": r.language.primary if r.language else "",
                    "sentiment": r.sentiment,
                    "emotions": r.emotions,
                    "keywords": r.keywords,
                    "psycholinguistic": r.psycholinguistic.__dict__ if r.psycholinguistic else {},
                    "quality_score": r.quality_score,
                }
            )
        formats = self.cfg.get("export", {}).get("formats", ["csv", "json", "parquet"])
        paths = export_features(rows, self.paths["processed"] / "exports", formats)
        stats = compute_statistics(results)
        report_dir = self.paths["reports"]
        report_dir.mkdir(parents=True, exist_ok=True)
        stats_path = report_dir / "text_statistics.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        paths["statistics"] = stats_path
        viz = generate_visualizations(results, report_dir / "visualizations")
        if viz:
            paths["visualizations"] = viz[0]
        return paths


def _process_worker(file_path: str, participant_id: str, session_id: str | None, split: str | None) -> TextPipelineResult:
    return TextPipeline().process_file(Path(file_path), participant_id, session_id, split)
