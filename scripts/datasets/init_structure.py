#!/usr/bin/env python3
"""Initialize datasets/ directory tree with README stubs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

STRUCTURE: dict[str, str] = {
    "datasets/README.md": "Root dataset directory for MindGraph++ pipeline.",
    "datasets/raw/README.md": "Immutable raw datasets. Never modify after download.",
    "datasets/raw/daic_woz/README.md": "DAIC-WOZ participant folders (300_P, ...). Owner: DaicWozAdapter.",
    "datasets/raw/dvlog/README.md": "D-VLOG labels.csv, acoustic.npy, visual.npy. Owner: DvlogAdapter.",
    "datasets/raw/future/README.md": "Placeholder for AVEC, MODMA, EATD, and future datasets.",
    "datasets/downloads/README.md": "Resumable download staging area.",
    "datasets/processed/README.md": "Processed features — each stage stored separately.",
    "datasets/processed/audio_features/README.md": "Step 5 audio outputs. Owner: processors.audio.",
    "datasets/processed/visual_features/README.md": "Step 6 video outputs. Owner: processors.video.",
    "datasets/processed/text_embeddings/README.md": "Step 7 transcript outputs. Owner: processors.transcript.",
    "datasets/processed/image_embeddings/README.md": "Image embedding outputs. Owner: configs/image.yaml.",
    "datasets/processed/fusion/README.md": "Step 8 fused features. Owner: processors.features.",
    "datasets/processed/normalized/README.md": "Step 10 normalized features. Owner: processors.normalization.",
    "datasets/processed/graphs/README.md": "Graph structures for MindGraph (future stage).",
    "datasets/processed/embeddings/README.md": "Step 9 embedding outputs. Owner: processors.embeddings.",
    "datasets/cache/README.md": "Versioned feature caches (wav2vec, bert, mediapipe, ...).",
    "datasets/cache/wav2vec/README.md": "Wav2Vec cache. Owner: configs/cache.yaml.",
    "datasets/cache/hubert/README.md": "HuBERT cache.",
    "datasets/cache/bert/README.md": "BERT / XLM-R cache.",
    "datasets/cache/indicbert/README.md": "IndicBERT cache.",
    "datasets/cache/mediapipe/README.md": "MediaPipe face landmark cache.",
    "datasets/cache/frames/README.md": "Extracted video frames cache.",
    "datasets/cache/transcripts/README.md": "Normalized transcript cache.",
    "datasets/cache/audio/README.md": "Intermediate audio cache.",
    "datasets/cache/video/README.md": "Intermediate video cache.",
    "datasets/cache/images/README.md": "Image cache.",
    "datasets/statistics/README.md": "Step 11 dataset statistics. Owner: processors.statistics.",
    "datasets/metadata/README.md": "Step 3 metadata CSVs and manifest.json.",
    "datasets/validation/README.md": "Step 2 validation and Step 13 verification reports.",
    "datasets/exports/README.md": "Step 12 multi-format exports.",
    "datasets/exports/csv/README.md": "CSV exports.",
    "datasets/exports/json/README.md": "JSON exports.",
    "datasets/exports/parquet/README.md": "Parquet exports.",
    "datasets/exports/pickle/README.md": "Pickle exports.",
    "datasets/exports/torch/README.md": "PyTorch tensor exports.",
    "datasets/exports/onnx/README.md": "ONNX exports (future).",
    "datasets/exports/statistics/README.md": "Export-time statistics.",
    "datasets/exports/reports/README.md": "Export reports.",
    "datasets/splits/README.md": "Official train/dev/test split copies.",
    "datasets/logs/README.md": "dataset.log, validation.log, processing.log, quality.log.",
    "datasets/configs/README.md": "Dataset-local config overrides (optional).",
}


def _readme_body(purpose: str, owner: str) -> str:
    return f"""# Dataset Directory

## Purpose
{purpose}

## Expected Files
See parent pipeline documentation and `configs/dataset.yaml`.

## Generation Process
Produced by `ml_pipeline.datasets` pipeline or `scripts/datasets/run_pipeline.py`.

## Owner Module
{owner}
"""


def main() -> None:
    for rel_path, purpose in STRUCTURE.items():
        path = ROOT / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        owner = purpose.split("Owner:")[-1].strip().rstrip(".") if "Owner:" in purpose else "ml_pipeline.datasets"
        if not path.exists():
            path.write_text(_readme_body(purpose, owner), encoding="utf-8")
        gitkeep = path.parent / ".gitkeep"
        if path.name != "README.md" and not any(path.parent.iterdir()):
            gitkeep.touch()


if __name__ == "__main__":
    main()
