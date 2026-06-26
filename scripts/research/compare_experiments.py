#!/usr/bin/env python3
"""Compare experiment metrics and generate comparison tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

COMPARISONS = [
    "baseline",
    "audio_only",
    "video_only",
    "text_only",
    "fusion",
    "fusion_graph",
    "fusion_graph_temporal",
]


def load_metrics(exp_dir: Path) -> dict:
    path = exp_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    experiments_dir = ROOT / "experiments"
    rows: list[str] = ["| Experiment | Model | Val F1 | Test F1 | ROC-AUC |", "|---|---|---|---|---|"]

    for exp_dir in sorted(experiments_dir.glob("EXP*")):
        metrics = load_metrics(exp_dir)
        if not metrics:
            continue
        val = metrics.get("validation_metrics", {})
        test = metrics.get("test_metrics", {})
        rows.append(
            f"| {metrics.get('experiment_id', exp_dir.name)} "
            f"| {metrics.get('model_version', 'n/a')} "
            f"| {val.get('f1', 'n/a')} "
            f"| {test.get('f1', 'n/a')} "
            f"| {test.get('roc_auc', 'n/a')} |"
        )

    output = ROOT / "research" / "comparisons" / "experiment_comparison.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    body = "# Experiment Comparison\n\n" + "\n".join(rows) + "\n"
    output.write_text(body, encoding="utf-8")
    print(f"Comparison table written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
