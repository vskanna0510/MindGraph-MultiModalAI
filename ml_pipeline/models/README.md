# MindGraph++ Multimodal Model Architecture

Research-grade modular multimodal AI (MP3 Part 1).

## Pipeline

```
Audio/Visual/Text/Image Encoders
  → Feature Projection (shared latent)
  → Cross-Modal Transformer
  → Temporal Fusion
  → Knowledge Graph Injection
  → GraphSAGE
  → Prediction + Risk Heads
  → Explainability Engine
```

## Usage

```python
from ml_pipeline.models import MindGraphMultimodal, MultimodalBatch

model = MindGraphMultimodal()
batch = MultimodalBatch.from_collated(collated_dict)
out = model.predict(batch)
```

```bash
make models-init
make models-validate
python scripts/models/validate_model.py --benchmark
```

## Configuration

`configs/model/*.yaml`

## Registry

MODEL_001–MODEL_010 tracked in `ml_pipeline/models/registry.json`
