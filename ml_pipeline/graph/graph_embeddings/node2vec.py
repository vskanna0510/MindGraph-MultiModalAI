"""Graph embedding algorithms."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np

from ml_pipeline.graph.schema.types import GraphSnapshot


def _build_adjacency(snapshot: GraphSnapshot) -> tuple[list[str], dict[str, list[str]]]:
    nodes = [n.node_id for n in snapshot.nodes]
    adj: dict[str, list[str]] = defaultdict(list)
    for e in snapshot.edges:
        adj[e.source_id].append(e.target_id)
        adj[e.target_id].append(e.source_id)
    return nodes, adj


def random_walk(snapshot: GraphSnapshot, start: str, length: int = 10) -> list[str]:
    _, adj = _build_adjacency(snapshot)
    walk = [start]
    for _ in range(length - 1):
        nbrs = adj.get(walk[-1], [])
        if not nbrs:
            break
        walk.append(random.choice(nbrs))
    return walk


def node2vec_embed(snapshot: GraphSnapshot, dim: int = 64, walks: int = 20, walk_length: int = 10) -> dict[str, np.ndarray]:
    nodes, _ = _build_adjacency(snapshot)
    if not nodes:
        return {}
    cooccur: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for node in nodes:
        for _ in range(walks):
            path = random_walk(snapshot, node, walk_length)
            for i, a in enumerate(path):
                for b in path[i + 1 : i + 3]:
                    cooccur[a][b] += 1.0
                    cooccur[b][a] += 1.0
    embeddings: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)
    for node in nodes:
        vec = rng.standard_normal(dim).astype(np.float32)
        for nbr, w in cooccur[node].items():
            if nbr in embeddings:
                vec += w * embeddings[nbr]
        norm = np.linalg.norm(vec) + 1e-8
        embeddings[node] = (vec / norm).astype(np.float32)
    return embeddings


def fastrp_embed(snapshot: GraphSnapshot, dim: int = 64, iterations: int = 3) -> dict[str, np.ndarray]:
    """FastRP-style random projection (offline approximation)."""
    nodes, adj = _build_adjacency(snapshot)
    n = len(nodes)
    if n == 0:
        return {}
    idx = {node: i for i, node in enumerate(nodes)}
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    for _ in range(iterations):
        new_emb = emb.copy()
        for i, node in enumerate(nodes):
            for nbr in adj[node]:
                j = idx[nbr]
                new_emb[i] += 0.5 * emb[j]
        norms = np.linalg.norm(new_emb, axis=1, keepdims=True) + 1e-8
        emb = (new_emb / norms).astype(np.float32)
    return {nodes[i]: emb[i] for i in range(n)}


def graph_embedding(snapshot: GraphSnapshot, algorithm: str = "node2vec", dim: int = 64) -> dict[str, np.ndarray]:
    if algorithm == "fastrp":
        return fastrp_embed(snapshot, dim)
    return node2vec_embed(snapshot, dim)
