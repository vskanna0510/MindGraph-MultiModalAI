# Evaluation Framework (MP3 Part 6)

Publication-quality model evaluation, explainability, and validation.

## Pipeline

```
Checkpoint → Validation → Testing → Calibration → Explainability → Ablation → Report → IEEE Figures
```

## Usage

```bash
make evaluate
python -m ml_pipeline.evaluation.run_evaluation --split test
python -m ml_pipeline.evaluation.run_evaluation --ablation
```

## Components

| Module | Purpose |
|--------|---------|
| `metrics/classification.py` | Full classification metric suite |
| `calibration/ece.py` | ECE, MCE, temperature scaling |
| `confusion/`, `roc/`, `pr/` | Publication figures |
| `explainability/` | Modality contribution, embeddings |
| `ablation/` | 15 mandatory ablation variants |
| `evaluator.py` | `ModelEvaluator` orchestrator |
| `reports/` | Automated research report |

## Tests

```bash
pytest ml_pipeline/evaluation/tests/test_evaluation.py -v
```
