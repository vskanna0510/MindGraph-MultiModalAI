# MindGraph++ Datasets

Dataset adapters for reproducible ML experiments. **Never commit raw patient data.**

## Supported Datasets

| Dataset | Status | License Required |
|---------|--------|------------------|
| DAIC-WOZ | Planned | Yes |
| D-VLOG | Planned | Yes |
| AVEC | Future | Yes |
| E-DAIC | Future | Yes |
| MODMA | Future | Yes |
| Custom Indian | Future | Yes |

## Usage

1. Obtain dataset from official source.
2. Place under `datasets/raw/<dataset_name>/` (gitignored).
3. Run preprocessing via `ml_pipeline/datasets/` adapters.
4. Processed outputs go to `datasets/processed/` (gitignored).

## Reproducibility

Record `dataset_version` in every experiment manifest under `experiments/`.
