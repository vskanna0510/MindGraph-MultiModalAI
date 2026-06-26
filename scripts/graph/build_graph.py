#!/usr/bin/env python3
"""Build temporal knowledge graphs from feature store sessions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.graph.builders import build_graph_builder
from ml_pipeline.graph.config import graph_paths
from ml_pipeline.graph.export.exporters import export_json
from ml_pipeline.graph.pipeline import GraphPipeline


def _load_sessions(limit: int) -> list[dict]:
    import pandas as pd

    path = ROOT / "datasets" / "processed" / "feature_store" / "metadata" / "sessions.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal knowledge graphs")
    parser.add_argument("--builder", default="temporal", choices=["neo4j", "temporal", "clinical", "research"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = graph_paths()
    index = _load_sessions(args.limit)
    pipe = GraphPipeline(builder_name=args.builder)
    count = 0
    for row in index:
        if args.limit and count >= args.limit:
            break
        data = {
            "participant_id": row.get("participant_id", row.get("session_id", "unknown")),
            "session_id": row.get("session_id", "unknown"),
            "dataset": row.get("dataset", "daic"),
            "language": row.get("language", "en"),
            "label": row.get("label", ""),
            "quality_score": float(row.get("quality_overall", 0.8) or 0.8),
            "modalities": {
                "audio": bool(row.get("has_audio", True)),
                "text": bool(row.get("has_text", True)),
                "visual": bool(row.get("has_visual", False)),
            },
        }
        snap = pipe.build_session_graph(data)
        export_json(snap, paths["exports"] / f"{data['session_id']}.json")
        count += 1
    print(f"Built {count} graphs → {paths['exports']}")


if __name__ == "__main__":
    main()
