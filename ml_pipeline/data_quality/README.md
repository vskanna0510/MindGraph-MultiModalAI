# MindGraph++ Data Quality Engine

Research validation and automated reporting (MP2 Part 7).

## Governance pipeline

1. Integrity validation
2. Metadata / consistency
3. Quality assessment
4. Feature verification
5. Embedding verification
6. Statistical validation
7. Leakage detection
8. Export validation
9. Research approval

## Usage

```bash
make data-quality-init
make data-quality-validate
make data-quality-ci   # exits non-zero on failure
```

## Outputs

`datasets/reports/data_quality/`

- `dataset_quality_report.md` / `.html`
- `integrity_report.md`
- `data_leakage_report.md`
- `embedding_validation_report.md`
- `statistics_report.json`
- `dataset_passport.json`
- `ci_validation_report.json`
- `audit_log.jsonl`

## Configuration

`configs/data_quality.yaml`
