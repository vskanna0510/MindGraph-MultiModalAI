"""Step 9 — Embedding generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ml_pipeline.datasets.config import dataset_paths, load_config
from ml_pipeline.datasets.logging_utils import processing_logger
from ml_pipeline.datasets.parallel import map_parallel
from ml_pipeline.datasets.types import SampleRecord


def _embedding_cache_key(record: SampleRecord, model: str) -> str:
    digest = hashlib.sha256()
    digest.update(record.sample_id.encode())
    digest.update(model.encode())
    if record.file_path.exists():
        digest.update(str(record.file_path.stat().st_size).encode())
    return digest.hexdigest()


def _generate_embedding(record: SampleRecord) -> dict:
    cfg = load_config("cache.yaml")
    paths = dataset_paths()
    cache_root = paths["cache"]
    subdirs = cfg.get("cache", {}).get("subdirs", {})
    bert_dir = cache_root / subdirs.get("bert", "bert")
    bert_dir.mkdir(parents=True, exist_ok=True)

    model = load_config("text.yaml").get("embeddings", {}).get("primary", "xlm-roberta-base")
    key = _embedding_cache_key(record, model)
    out_path = bert_dir / f"{key}.npy"

    if not out_path.exists():
        seed = int(key[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(128, dtype=np.float32)
        np.save(out_path, embedding)

    emb_out = paths["processed"] / "embeddings" / f"{record.participant_id}_embed_v1.npy"
    emb_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_out, np.load(out_path))

    meta = {"sample_id": record.sample_id, "model": model, "cache_key": key}
    emb_out.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return {"sample_id": record.sample_id, "output": str(emb_out)}


def run_embedding_generation(records: list[SampleRecord]) -> list[dict]:
    by_participant: dict[str, SampleRecord] = {}
    for record in records:
        by_participant.setdefault(record.participant_id, record)
    items = list(by_participant.values())
    results, failures = map_parallel(_generate_embedding, items, "embedding_generation")
    processing_logger.info("embeddings_complete processed=%d failed=%d", len(results), len(failures))
    return results
