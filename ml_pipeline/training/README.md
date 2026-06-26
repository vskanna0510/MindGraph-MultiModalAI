# Training Framework (MP3 Part 5)

Publication-grade training for stable, explainable, reproducible depression detection.

## Pipeline

```
Dataset → DataLoader → Encoders → Fusion → Temporal → Graph → GNN → MultiTaskHeads → Loss → Optimizer → Checkpoint
```

## Multi-Task Heads

- Classification, regression, confidence, temporal (7/30/90-day), graph refinement

## Loss

`MultiTaskLoss` = classification + regression + graph + temporal + consistency + regularization

Configurable: focal, label smoothing, Huber/MAE/MSE, class weights.

## Training

```bash
make train
python -m ml_pipeline.training.train --dry-run
python -m ml_pipeline.training.train --epochs 10 --batch-size 8
```

## Stages

1. Encoder warm-up → 2. Fusion → 3. Temporal → 4. Graph → 5. End-to-end fine-tuning

## Tests

```bash
pytest ml_pipeline/training/tests/test_training.py -v
```
