"""Graph visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ml_pipeline.graph.analytics.metrics import compute_analytics
from ml_pipeline.graph.schema.types import GraphSnapshot


def export_graph_report(snapshot: GraphSnapshot, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    analytics = compute_analytics(snapshot)
    path = output_dir / "graph_analytics.json"
    path.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
    return [path]
