# ML Pipeline

Training, evaluation, feature extraction, and model deployment for MindGraph++.

## Modules

| Directory | Purpose |
|-----------|---------|
| `feature_extractors/` | Audio, visual, text, image pipelines |
| `fusion/` | Cross-modal transformer fusion |
| `training/` | Training loops and checkpoints |
| `evaluation/` | Metrics and ablation runners |
| `explainability/` | SHAP, attention, graph path exports |
| `deployment/onnx` | ONNX export for edge |
| `deployment/tflite` | TFLite export for mobile |

## Install ML Dependencies

```bash
pip install -r requirements-ml.txt
```

## Training

```bash
make train
```

Configuration: `configs/training.yaml`
