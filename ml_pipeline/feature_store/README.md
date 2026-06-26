# MindGraph++ Feature Store

Multimodal feature platform — single source of truth for training.

## Principles

- Never train on raw files directly
- Every tensor is traceable (feature ID, version, checksum)
- Never overwrite old feature versions
- Missing modalities use masks, not sample discard

## Usage

```bash
make feature-store-init
make feature-store-build
python scripts/feature_store/run_feature_store.py --limit 20
```

## Architecture

```
FeatureStorePipeline
  → build_index
  → validation
  → statistics
  → visualizations
  → sync / outlier reports
  → export
```

## Datasets

- `DAICDataset` / `DVLOGDataset` / `CombinedDataset`
- `collate_multimodal_batch` for PyTorch DataLoader

## Configuration

`configs/feature_store.yaml`
