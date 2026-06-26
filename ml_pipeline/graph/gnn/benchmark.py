"""Benchmark GNN architectures."""

from __future__ import annotations

import time
from typing import Any

import torch

from ml_pipeline.graph.gnn.registry import GNN_REGISTRY, build_gnn


def benchmark_gnns(d_model: int = 256, batch_size: int = 4, runs: int = 5) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    x = torch.randn(batch_size, d_model)
    adj = torch.eye(batch_size)
    for name in GNN_REGISTRY:
        gnn = build_gnn(name, d_model, {"hidden_dim": 128, "num_layers": 1, "heads": 2})
        gnn.eval()
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                try:
                    gnn(x, adj)
                except TypeError:
                    gnn(x, adj, torch.zeros(batch_size))
                times.append((time.perf_counter() - start) * 1000)
        params = sum(p.numel() for p in gnn.parameters())
        results[name] = {"mean_ms": sum(times) / len(times), "params": params}
    return results
