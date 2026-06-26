# Model Card

## Overview
MindGraph++ multimodal depression detection architecture
## Architecture
Independent encoders + cross-modal transformer + GNN
## Input
{'modalities': ['audio', 'visual', 'text', 'image']}
## Output
{'logits': '[B, 2]', 'risk_logits': '[B, 3]'}
## Training Dataset
DAIC-WOZ, D-VLOG
## Metrics
{}
## Known Limitations

## Hardware
CUDA / CPU
## Inference Time
0.00 ms
## Memory
0.0 MB
## Export Formats
['pytorch', 'torchscript', 'onnx']
