"""Graph analytics metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ml_pipeline.graph.schema.types import GraphSnapshot


def node_degree(snapshot: GraphSnapshot) -> dict[str, int]:
    degree: dict[str, int] = defaultdict(int)
    for e in snapshot.edges:
        degree[e.source_id] += 1
        degree[e.target_id] += 1
    return dict(degree)


def connected_components(snapshot: GraphSnapshot) -> list[set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    nodes = {n.node_id for n in snapshot.nodes}
    for e in snapshot.edges:
        adj[e.source_id].add(e.target_id)
        adj[e.target_id].add(e.source_id)
    visited: set[str] = set()
    components: list[set[str]] = []
    for node in nodes:
        if node in visited:
            continue
        stack = [node]
        comp: set[str] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            stack.extend(adj[cur] - visited)
        components.append(comp)
    return components


def temporal_density(snapshot: GraphSnapshot) -> float:
    temporal = sum(1 for e in snapshot.edges if e.rel_type in ("TEMPORALLY_PRECEDES", "FOLLOWED_BY"))
    sessions = sum(1 for n in snapshot.nodes if n.label == "Session")
    return temporal / max(sessions - 1, 1)


def compute_analytics(snapshot: GraphSnapshot) -> dict[str, Any]:
    degree = node_degree(snapshot)
    components = connected_components(snapshot)
    return {
        "node_count": snapshot.node_count,
        "edge_count": snapshot.edge_count,
        "mean_degree": sum(degree.values()) / max(len(degree), 1),
        "num_components": len(components),
        "temporal_density": temporal_density(snapshot),
        "max_degree": max(degree.values()) if degree else 0,
    }
