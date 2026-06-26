"""Graph export formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml_pipeline.graph.schema.types import GraphSnapshot


def export_json(snapshot: GraphSnapshot, path: Path) -> Path:
    data = {
        "version": snapshot.version,
        "snapshot_id": snapshot.snapshot_id,
        "nodes": [{"id": n.node_id, "label": n.label, "properties": n.properties} for n in snapshot.nodes],
        "edges": [{"source": e.source_id, "target": e.target_id, "type": e.rel_type, "properties": e.properties} for e in snapshot.edges],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def export_csv(snapshot: GraphSnapshot, dir_path: Path) -> dict[str, Path]:
    dir_path.mkdir(parents=True, exist_ok=True)
    nodes_path = dir_path / "nodes.csv"
    edges_path = dir_path / "edges.csv"
    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "label"])
        for n in snapshot.nodes:
            w.writerow([n.node_id, n.label])
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "type"])
        for e in snapshot.edges:
            w.writerow([e.source_id, e.target_id, e.rel_type])
    return {"nodes": nodes_path, "edges": edges_path}


def export_graphml(snapshot: GraphSnapshot, path: Path) -> Path:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">', '<graph edgedefault="directed">']
    for n in snapshot.nodes:
        lines.append(f'<node id="{n.node_id}"><data key="label">{n.label}</data></node>')
    for i, e in enumerate(snapshot.edges):
        lines.append(f'<edge id="e{i}" source="{e.source_id}" target="{e.target_id}"><data key="type">{e.rel_type}</data></edge>')
    lines.extend(["</graph>", "</graphml>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def to_pyg_dict(snapshot: GraphSnapshot, node_features: np.ndarray | None = None) -> dict[str, Any]:
    node_ids = [n.node_id for n in snapshot.nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = [[idx[e.source_id], idx[e.target_id]] for e in snapshot.edges if e.source_id in idx and e.target_id in idx]
    if node_features is None:
        node_features = np.random.randn(len(node_ids), 16).astype(np.float32)
    return {
        "x": node_features,
        "edge_index": np.array(edge_index, dtype=np.int64).T if edge_index else np.zeros((2, 0), dtype=np.int64),
        "num_nodes": len(node_ids),
        "node_ids": node_ids,
    }
