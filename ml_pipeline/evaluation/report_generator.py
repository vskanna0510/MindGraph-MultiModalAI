"""Automated research report generation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def generate_experiment_report(
    experiment_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    """Generate a Markdown research report from an experiment directory."""
    metrics_path = experiment_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {experiment_dir}")

    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    exp_id = metrics.get("experiment_id", experiment_dir.name)
    output = output_dir or Path("reports/experiment")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{exp_id}_report.md"

    validation = metrics.get("validation_metrics", {})
    test = metrics.get("test_metrics", {})
    env = metrics.get("environment", {})

    lines = [
        f"# Experiment Report — {exp_id}",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Objective",
        "See experiment README for research motivation.",
        "",
        "## Configuration",
        f"- Git commit: `{metrics.get('git_commit', 'unknown')}`",
        f"- Dataset version: {metrics.get('dataset_version', 'unknown')}",
        f"- Model version: {metrics.get('model_version', 'unknown')}",
        f"- Config path: `{metrics.get('config_path', '')}`",
        "",
        "## Method",
        "Refer to `hyperparameters.yaml` and `config.yaml` in the experiment directory.",
        "",
        "## Environment",
        f"- OS: {env.get('operating_system', 'unknown')}",
        f"- Python: {env.get('python_version', 'unknown')}",
        f"- PyTorch: {env.get('pytorch_version', 'n/a')}",
        f"- CUDA: {env.get('cuda_version', 'n/a')}",
        "",
        "## Results",
        "### Validation",
        "```json",
        json.dumps(validation, indent=2),
        "```",
        "### Test",
        "```json",
        json.dumps(test, indent=2),
        "```",
        "",
        "## Observations",
        "Document qualitative findings.",
        "",
        "## Limitations",
        "Document limitations discovered during this run.",
        "",
        "## Future Improvements",
        "Define follow-up experiments.",
        "",
        "## Reproducibility",
        f"```bash",
        f"python scripts/research/run_experiment.py --experiment-id {exp_id}",
        f"```",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
