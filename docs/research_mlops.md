# Research & MLOps Guide

Master Prompt 1 Part 4 — reproducible research engineering.

## One-Command Reproduction

```bash
python scripts/research/run_experiment.py --config configs/ml/baseline.yaml --experiment-id EXP001 --init-only
```

## Experiment Tracking

Every experiment folder under `experiments/EXP###/` contains:

- `config.yaml`, `hyperparameters.yaml`, `metrics.json`, `manifest.json`
- `seeds.json`, `training.log`, `validation.csv`, `test.csv`
- `README.md` (objective, results, limitations)

**Never overwrite** existing experiment directories.

## Randomness Control

`ml_pipeline/utils/reproducibility.py` sets Python, NumPy, PyTorch, CUDA, DataLoader, and hash seeds.

## Statistical Analysis

`ml_pipeline/evaluation/statistics.py` — mean, median, CI, paired t-test, Wilcoxon, McNemar.

## Comparison Tables

```bash
python scripts/research/compare_experiments.py
```

Output: `research/comparisons/experiment_comparison.md`

## Data Versioning

```
raw → cleaned → processed → feature cache → training ready
```

Validate datasets:

```python
from ml_pipeline.datasets.validation import validate_dataset_root, write_validation_report
```

## IEEE Figure Standards

Configured in `configs/ml/evaluation.yaml`: 300 DPI, PNG + SVG exports.
