# MindGraph++ Research Engineering & MLOps

This directory contains the **reproducible research layer** for IEEE/thesis-ready experimentation.

## Principles

- Every experiment is immutable (never overwrite `EXP###` folders).
- Every run records git commit, seeds, hardware, and configuration.
- One command reproduction: `python scripts/research/run_experiment.py --config configs/ml/baseline.yaml`

## Structure

| Directory | Purpose |
|-----------|---------|
| `literature/` | Paper notes and citations |
| `experiments/` | Research experiment copies |
| `ablations/` | Ablation study artifacts |
| `benchmarks/` | Performance benchmarks |
| `reproducibility/` | Seed bundles and manifests |
| `publications/` | Camera-ready figures and tables |
| `supplementary/` | Supplementary materials |

## Workflow

```bash
python scripts/research/init_research_structure.py
python scripts/research/run_experiment.py --config configs/ml/audio.yaml --init-only
python scripts/research/compare_experiments.py
```

## IEEE Readiness

- Figures: 300 DPI minimum (`configs/ml/evaluation.yaml`)
- Tables: auto-generated in `reports/experiment/`
- Statistical tests: `ml_pipeline/evaluation/statistics.py`
