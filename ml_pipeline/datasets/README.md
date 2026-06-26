# Dataset Pipeline (`ml_pipeline.datasets`)

Production-grade, adapter-based dataset preparation for multimodal depression detection research.

## Architecture

- **Adapters** (`adapters/`): One class per dataset — `DaicWozAdapter`, `DvlogAdapter`. Add future datasets (AVEC, MODMA, EATD) by implementing `BaseDatasetAdapter`.
- **Pipeline** (`pipeline.py`): Executes 13 steps in fixed order (never reorder).
- **Configs** (`configs/dataset.yaml`, `audio.yaml`, `video.yaml`, …): No hardcoded paths or label mappings.

## Pipeline Steps

1. Discovery → `dataset_index.csv`
2. Validation → `datasets/validation/validation_report.json`
3. Metadata → `datasets/metadata/*.csv`
4. Split verification → `splits.csv`
5. Audio processing
6. Video processing
7. Transcript processing
8. Feature extraction
9. Embedding generation
10. Normalization
11. Statistics
12. Export (CSV, JSON, pickle, torch)
13. Verification

## Usage

```bash
# Initialize directory tree + README stubs
make datasets-init

# Run full pipeline (after placing raw data)
make datasets-pipeline

# Or directly
python scripts/datasets/run_pipeline.py --init-structure
python scripts/datasets/download_daic.py
python scripts/datasets/download_dvlog.py
python scripts/datasets/verify_download.py datasets/raw/daic_woz
```

## Raw Data (Immutable)

- `datasets/raw/daic_woz/{participant}_P/`
- `datasets/raw/dvlog/labels.csv`, `acoustic.npy`, `visual.npy`

Never modify raw files. Processed outputs go to `datasets/processed/`.

## Environment

- `MINDGRAPH_DATASETS_ROOT` — override dataset root (used in tests)
- `DAIC_DOWNLOAD_URL` / `DVLOG_DOWNLOAD_URL` — optional download URLs

## Logs

- `datasets/logs/dataset.log`
- `datasets/logs/validation.log`
- `datasets/logs/processing.log`
- `datasets/logs/quality.log`
