# Model Card Template

## Model Details
- **Version:** MODEL_XXX_NAME
- **Date:** YYYY-MM-DD
- **Config:** configs/ml/<name>.yaml
- **Experiment:** EXP###

## Intended Use
Wellness screening research only. Not a medical diagnosis device.

## Training Data
- Dataset:
- Version:
- Splits:

## Metrics
| Split | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Val   |          |     |         |
| Test  |          |     |         |

## Limitations
-

## Ethical Considerations
-

## Reproducibility
```bash
python scripts/research/run_experiment.py --config configs/ml/<name>.yaml --experiment-id EXP###
```
