"""Automated IEEE-style research report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def generate_research_report(
    eval_results: dict[str, Any],
    output_dir: Path,
    experiment_meta: dict[str, Any] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = experiment_meta or {}
    m = eval_results.get("metrics", {})
    errors = eval_results.get("error_analysis", {})
    checklist = eval_results.get("checklist", {})

    lines = [
        "# MindGraph++ Research Evaluation Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Executive Summary",
        f"- **F1**: {m.get('f1', 0):.4f}",
        f"- **ROC AUC**: {m.get('roc_auc', 0):.4f}",
        f"- **ECE**: {m.get('ece', 0):.4f}",
        f"- **Balanced Accuracy**: {m.get('balanced_accuracy', 0):.4f}",
        "",
        "## Methodology",
        "Multimodal fusion → temporal learning → knowledge graph → GNN → multi-task heads.",
        "",
        "## Dataset",
        f"- Version: {meta.get('dataset_version', 'feature_store v1')}",
        f"- Split: {eval_results.get('split', 'test')}",
        "",
        "## Results",
        "### Primary Metrics",
        "```json",
        json.dumps(m, indent=2),
        "```",
        "",
        "## Calibration",
        f"Expected Calibration Error: **{m.get('ece', 0):.4f}**",
        "",
        "## Error Analysis",
        f"- False positives: {errors.get('false_positives', 0)}",
        f"- False negatives: {errors.get('false_negatives', 0)}",
        "",
        "### Recommendations",
    ]
    for rec in errors.get("recommendations", []):
        lines.append(f"- {rec}")

    lines.extend([
        "",
        "## Statistical Analysis",
        "Run paired tests via `ml_pipeline.evaluation.statistics` for model comparisons.",
        "",
        "## Limitations",
        "- Single-dataset evaluation; external validation recommended.",
        "- Calibration on held-out split only.",
        "",
        "## Future Work",
        "- Cross-dataset generalization (DAIC → D-VLOG).",
        "- Clinician-in-the-loop explanation validation.",
        "",
        "## Publication Readiness Checklist",
    ])
    for k, v in checklist.items():
        lines.append(f"- [{'x' if v else ' '}] {k.replace('_', ' ').title()}")

    path = output_dir / "research_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    json_path = output_dir / "research_report.json"
    json_path.write_text(json.dumps({"metrics": m, "checklist": checklist, "meta": meta}, indent=2), encoding="utf-8")
    return path
