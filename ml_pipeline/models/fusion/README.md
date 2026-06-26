# Multimodal Fusion (MP3 Part 3)

Publication-grade fusion architecture for MindGraph++. Encoders and fusion are decoupled — every strategy is independently trainable, replaceable, and benchmarkable.

## Pipeline

```
Encoders → Projection → Fusion Strategy → Temporal Fusion → (Graph / Heads)
```

## Strategies

| Strategy | Module | Description |
|----------|--------|-------------|
| `early` | `early_fusion.py` | Concatenate projected embeddings |
| `late` | `late_fusion.py` | Modality-specific heads + ensemble |
| `cross_attention` | `cross_attention.py` | Self + cross attention pooling |
| `co_attention` | `co_attention.py` | Bidirectional co-attention |
| `transformer` / `cross_modal_transformer` | `transformer_fusion.py` | CLS + modality tokens + transformer |
| `gated` | `gated_fusion.py` | Learned modality gates |
| `dynamic` | `dynamic_fusion.py` | Gated fusion + GRU context |
| `residual` | `residual_fusion.py` | Transformer + residual pathways |
| `quality_aware` | `quality_aware.py` | Quality-weighted gated fusion |

## Usage

```python
from ml_pipeline.models.fusion import build_fusion, MultimodalFusionStack

stack = MultimodalFusionStack(config)  # from configs/model/fusion.yaml
out = stack(embeddings, presence, quality_scores)
fused = stack.fused_tensor(out)
```

## Configuration

See `configs/model/fusion.yaml` — strategy, heads, layers, positional encoding, temporal settings, and benchmark list.

## Benchmark

```bash
make models-benchmark-fusion
```

## Tests

```bash
pytest ml_pipeline/models/fusion/tests/test_fusion.py -v
```
