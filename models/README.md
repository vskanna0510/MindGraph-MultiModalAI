# Model Registry — MindGraph++

Versioned model artifacts for reproducible research. **Never overwrite** a registered model version.

## Layout

| Directory | Purpose |
|-----------|---------|
| `baseline/` | Baseline multimodal model |
| `audio/` | Audio-only encoders |
| `visual/` | Video/visual encoders |
| `text/` | Text/multilingual encoders |
| `fusion/` | Fusion architectures |
| `graph/` | GNN + temporal graph models |
| `best/` | Production-candidate checkpoints |
| `archive/` | Retired versions |
| `experimental/` | Unvalidated research models |

## Versioning Convention

```
MODEL_001_BASELINE
MODEL_002_AUDIO
MODEL_003_VISUAL
MODEL_004_TEXT
MODEL_005_FUSION
MODEL_006_GRAPH
MODEL_007_FINAL
```

## Required Per Model

- `README.md` — purpose and training context
- `model_card.md` — metrics, limitations, ethical notes
- `training_config.yaml` — frozen hyperparameters
- `metrics.json` — validation/test metrics
- `checkpoint.pt` — PyTorch weights (gitignored)
- `exports/` — ONNX, TorchScript, TFLite (gitignored)

## Checkpoint Strategy

Save: best validation, best F1, best loss, last epoch, every N epochs.  
Support resume via experiment manifest `checkpoint_path`.
