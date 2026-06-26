# ML Experiment Configurations

Hyperparameters are **separated by concern** in every file:

| Section | Contents |
|---------|----------|
| `experiment` | ID, name, tags |
| `model` | Architecture parameters |
| `training` | Epochs, batch size, seeds |
| `dataset` | Paths, splits, versions |
| `optimization` | Optimizer, LR, scheduler |
| `evaluation` | Metrics, thresholds |
| `deployment` | Export formats |
| `reproducibility` | Deterministic flags |

## Files

| Config | Purpose |
|--------|---------|
| `baseline.yaml` | Multimodal baseline |
| `audio.yaml` | Audio-only ablation |
| `video.yaml` | Visual-only ablation |
| `text.yaml` | Text-only ablation |
| `fusion.yaml` | Cross-modal fusion |
| `evaluation.yaml` | IEEE evaluation protocol |

## Usage

```bash
python scripts/research/run_experiment.py --config configs/ml/baseline.yaml --init-only
```
