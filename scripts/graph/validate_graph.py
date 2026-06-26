#!/usr/bin/env python3
"""Validate graph integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_pipeline.graph.evaluation.validator import GraphValidator
from ml_pipeline.graph.schema.types import GraphEdge, GraphNode, GraphSnapshot


def load_snapshot(path: Path) -> GraphSnapshot:
    raw = json.loads(path.read_text(encoding="utf-8"))
    nodes = [GraphNode(n["id"], n["label"], n.get("properties", {})) for n in raw["nodes"]]
    edges = [GraphEdge(e["source"], e["target"], e["type"], e.get("properties", {})) for e in raw["edges"]]
    return GraphSnapshot(nodes=nodes, edges=edges, version=raw.get("version", "1.0.0"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate graph snapshots")
    parser.add_argument("path", type=Path, nargs="?", default=Path("datasets/processed/graph/exports"))
    args = parser.parse_args()
    paths = list(args.path.glob("*.json")) if args.path.is_dir() else [args.path]
    validator = GraphValidator()
    passed = 0
    for p in paths:
        snap = load_snapshot(p)
        ok, errs = validator.validate(snap)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {p.name}: {errs or 'ok'}")
        passed += int(ok)
    print(f"\n{passed}/{len(paths)} passed")


if __name__ == "__main__":
    main()
